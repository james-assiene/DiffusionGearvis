"""T-0211 group B, B5: the prefix property of live guidance v4, through the pinned Gemma 4 template
(docs/tasks/T-0211.md).

**What is checked** (T-0174 A5's rig and checker, reused as T-0191 A5 does): ``cli`` builds a live session over the
fakes (T-0152's ``_cli_run``: ``astra session --tools live --browser headless <flags>``, a real start stopped before
serving; web mode unless ``--no-web``) and the keywords it passed to ``Session`` build a rig session over the fake LLM
and a fake live browser. The rig runs two turns, the first with three executed ``click_at`` steps then an answer, the
second a plain answer. Then ``test_t0174_prefix_oracle.check_prefix_property``: every wire body and prefill, rendered
by the pinned template (``tests/cd/oracle/gemma4_template.py``), has one byte-identical system turn starting
``<bos><|turn>system\\n<|think|>``; every ``llm.request`` record carries that system text and ``thinking: true``; the
thinking keys and ``chat_template_kwargs`` are identical in every body; the system text ends with the guidance once.

- ``test_prefix_property_v4``: ``--live-guidance v4`` in web mode (the live defaults: wait tool on); the guidance is
  ``LIVE_GUIDANCE["v4"]`` plus the wait sentence.
- ``test_v4_v3_differ_only_by_sentences`` (both wait states): the v4 and v3 rendered system turns are equal except for
  ``" " + S_SHARED + " " + S_SOURCE`` inserted once, right after v3's text: with the wait tool on that is before
  ``" " + WAIT_TOOL_GUIDANCE`` (reviewer mutation 3: inserted after the wait sentence is caught), with
  ``--wait-tool off`` at the end of the system text.
- Adversarial ``test_switch_between_turns_caught`` (the failure named by the implementation notes: the text chosen per
  request, not once): a rig whose second-turn requests carry v3's text after a first turn with v4's is refused by the
  same checker; premise: both texts reached the wire, and the unswitched v4 rig passes the same checker.

S_SHARED, S_SOURCE and the wait sentence are typed from the tasks (T-0211 B1, T-0202 H-T202-4), never imported.

The oracle needs jinja2 (the ``cd`` dependency group) and the pinned template in the local HF cache; without them the
rows skip naming what is missing. Run with::

    uv run --all-extras --group cd pytest tests/contract/test_t0211_prefix_oracle.py -q -rxX

Every row was ``pending(task="T-0211")`` until activation (2026-10-30, 10a7d921): at BASE (175a29b0) ``LIVE_TOOL_GUIDANCE_V4`` does not exist and ``cli``
refuses ``--live-guidance v4``; each row fails first on the named ``AssertionError`` "... does not exist yet".

Regime: CPU over fakes, ``ScriptedClock``, thinking on, tools live, web mode, loop on, 1280x800 frames, the template
of ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev 2ea8372; n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra import config
from local_astra.conductor import prompt_builder as pb
from local_astra.contracts.llm import Message, Prompt, TextPart
from tests.contract import test_t0174_prefix_oracle as a5
from tests.unit import _t0202_pins as t0202
from tests.unit.test_session_live import _wire

FLAG = "--live-guidance"
V4 = [FLAG, "v4"]
V3 = [FLAG, "v3"]
WAIT_OFF = ["--wait-tool", "off"]
WAIT_ON = ["--wait-tool", "on"]
#: B1, typed from the task (never imported)
S_SHARED = (
    "A shared screen shows what the user means, but you act only in your own browser, which is always available: "
    "to do a web task from a shared screen, press Ctrl+L to open your browser's start page and continue there."
)
S_SOURCE = (
    "Before you answer with a fix or a fact you found through google_search, open the page that states it, read it "
    "there, and name that page in your answer."
)
INSERT = " " + S_SHARED + " " + S_SOURCE
#: H-T202-4, typed in ``_t0202_pins`` from T-0202
WAIT = t0202.SENTENCE


def v4_text() -> str:
    """The named failure at BASE: v4 is not a version and its text does not exist."""
    versions = getattr(config, "LIVE_GUIDANCE_VERSIONS", ())
    assert "v4" in versions, (
        f"T-0211 B3: 'v4' in config.LIVE_GUIDANCE_VERSIONS does not exist yet; {versions!r}"
    )
    value = getattr(pb, "LIVE_TOOL_GUIDANCE_V4", None)
    assert isinstance(value, str), (
        "T-0211 B1: prompt_builder.LIVE_TOOL_GUIDANCE_V4 does not exist yet"
    )
    assert pb.LIVE_GUIDANCE.get("v4") is value, (
        "T-0211 B3: LIVE_GUIDANCE['v4'] is LIVE_TOOL_GUIDANCE_V4"
    )
    assert len(S_SHARED) == 213 and len(S_SOURCE) == 151, (
        "premise: the sentences as typed from the task"
    )
    return value


def with_wait(text: str) -> str:
    """The guidance a live session carries with the wait tool on: the text, one space, H-T202-4's sentence."""
    return text + " " + WAIT


def swap_guidance(prompt: Prompt, current: str, replacement: str) -> Prompt:
    first = prompt.messages[0]
    assert first.role == "system"
    text = first.text()
    assert text.endswith(current), "premise: the rig's system text ends with the session's guidance"
    swapped = Message(role="system", content=(TextPart(text=text[: -len(current)] + replacement),))
    return prompt.model_copy(update={"messages": (swapped, *prompt.messages[1:])})


async def switching_turns(
    tmp: Path, kwargs: dict[str, Any], echo: str, first: str, second: str
) -> dict[str, Any]:
    """T-0191 A5's ``switching_turns`` pattern: T-0174 A5's two turns, with every request of the second turn carrying
    ``second`` in place of the session's guidance ``first`` (the version switched between turns)."""
    from tests.fakes.llm import is_caption_prompt
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_frame
    from tests.unit._tool_turn_support import answer, tool
    from tests.unit.test_t0117_differential import _PolicyMachine
    from tests.unit.test_t0134_label import next_turn

    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser(
        [live_frame("t0211-onset"), live_frame("t0211-turn2")],
        post_frames=[live_frame(f"t0211-after-{k}") for k in (1, 2, 3)],
        echo=echo,
    )
    switched = [False]
    async with feedback_rig(
        tmp,
        browser,
        script=[answer("I will add it. "), tool(a5.CLICKS[0])],
        session_kwargs=kwargs,
        machine_cls=_PolicyMachine,
    ) as rig:
        real = rig.llm.generate

        def generate(prompt: Prompt, *a: Any, **k: Any) -> Any:
            if switched[0] and not is_caption_prompt(prompt):
                prompt = swap_guidance(prompt, first, second)
            return real(prompt, *a, **k)

        rig.llm.generate = generate
        rig.llm.script_steps(
            [answer("Next. "), tool(a5.CLICKS[1])],
            [answer("Next. "), tool(a5.CLICKS[2])],
            [answer("The towel is in the cart. ")],
        )
        await rig.ask(a5.Q1)
        await tsl.until_steps(rig, 3, limit_ms=30_000, step=100)
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        switched[0] = True
        await next_turn(rig, 2, a5.Q2, [answer("The cart says one item. ")])
        bodies = [
            body
            for (_, prompt, _), body in zip(rig.llm.generates, rig.llm.bodies, strict=True)
            if not is_caption_prompt(prompt)
        ]
        prefills = [_wire(p) for _, p in rig.llm.prefills if not is_caption_prompt(p)]
        log_path = Path(rig.log.path)
    records = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
    return {
        "bodies": bodies,
        "prefills": prefills,
        "requests": [r["data"] for r in records if r["kind"] == "llm.request"],
    }


def test_prefix_property_v4(tmp_path: Path) -> None:
    """B5: a cli-built live web-mode v4 session (live defaults: the wait tool on): one rendered system turn across
    every request of two turns (the first with three tool steps), ``<|think|>`` first, the same thinking flags and
    ``chat_template_kwargs``, the system text ending with v4's guidance plus the wait sentence once."""
    v4 = v4_text()
    assert t0202.default_on(), "premise: the wait tool is on by default in live (T-0202 D)"
    tpl = a5._oracle()
    turn = a5.check_prefix_property(tpl, run := a5.run_rig(tmp_path, V4), with_wait(v4))
    assert turn.count(S_SHARED) == 1 and turn.count(S_SOURCE) == 1, (
        "T-0211 B5: each v4 sentence is in the system turn once"
    )
    texts = {r["prompt"]["messages"][0]["content"][0]["text"] for r in run["requests"]}
    assert len(texts) == 1 and next(iter(texts)).endswith(S_SOURCE + " " + WAIT), (
        "T-0211 B5: the logged system text ends with S_SOURCE then the wait sentence"
    )


@pytest.mark.parametrize("wait", ["on", "off"])
def test_v4_v3_differ_only_by_sentences(tmp_path: Path, wait: str) -> None:
    """B5: the v4 and v3 rendered system turns (web mode) differ only by ``" " + S_SHARED + " " + S_SOURCE``, inserted
    once right after v3's text: before ``" " + WAIT_TOOL_GUIDANCE`` with the wait tool on, at the end of the guidance
    (and of the system text) with ``--wait-tool off``."""
    v4 = v4_text()
    v3 = pb.LIVE_TOOL_GUIDANCE_V3
    assert v4 == v3 + INSERT, (
        "T-0211 B1: LIVE_TOOL_GUIDANCE_V4 is v3 + ' ' + S_SHARED + ' ' + S_SOURCE"
    )
    extra = WAIT_ON if wait == "on" else WAIT_OFF
    g3 = with_wait(v3) if wait == "on" else v3
    g4 = with_wait(v4) if wait == "on" else v4
    tpl = a5._oracle()
    t3 = a5.check_prefix_property(tpl, a5.run_rig(tmp_path / "v3", [*V3, *extra]), g3)
    t4 = a5.check_prefix_property(tpl, a5.run_rig(tmp_path / "v4", [*V4, *extra]), g4)
    assert t3.count(v3) == 1, "premise: v3 appears once in its system turn"
    assert (WAIT in t3) is (wait == "on"), (
        f"premise: the wait sentence is in v3's turn iff wait={wait}"
    )
    i = t3.index(v3) + len(v3)
    if wait == "on":
        assert t3[i:].startswith(" " + WAIT), "premise: the wait sentence follows v3's text"
    assert t4 == t3[:i] + INSERT + t3[i:], (
        "T-0211 B5: the v4 and v3 system turns differ other than by ' ' + S_SHARED + ' ' + S_SOURCE after v3's text"
    )
    if wait == "on":
        assert S_SOURCE + " " + WAIT in t4, (
            "T-0211 B5: the sentences go before the wait sentence, not after"
        )


def test_switch_between_turns_caught(tmp_path: Path) -> None:
    """B5 adversarial: a rig whose second turn carries v3's text after a first turn with v4's is refused by
    ``check_prefix_property``; premise: both texts reached the wire, and the unswitched v4 rig passes the same
    checker."""
    v4 = v4_text()
    v3 = pb.LIVE_TOOL_GUIDANCE_V3
    g4, g3 = with_wait(v4), with_wait(v3)
    tpl = a5._oracle()
    a5.check_prefix_property(tpl, a5.run_rig(tmp_path / "plain", V4), g4)
    kwargs, echo = a5.cli_kwargs(tmp_path / "cli", V4)
    got = asyncio.run(switching_turns(tmp_path / "rig", kwargs, echo, g4, g3))
    texts = [json.dumps(b["messages"][0]["content"]) for b in got["bodies"]]
    assert len(set(texts)) == 2, (
        f"premise: the switch put two system texts on the wire; {len(set(texts))}"
    )
    shared = json.dumps(S_SHARED)[1:-1]
    v3_wire = json.dumps(g3)[1:-1]
    assert any(shared in t for t in texts), "premise: v4 on the wire"
    assert any(shared not in t and v3_wire in t for t in texts), "premise: v3 alone on the wire"
    with pytest.raises(AssertionError, match="T-0174 A5: the rendered system turn varies"):
        a5.check_prefix_property(tpl, got, g4)


# --------------------------------------------------------------------------------------------- B-R8: the check


#: B-R8 (row 3): the turn-1 draft (a no-call answer after three executed clicks) and the check's own answer
CLAIM = "The towel is in the cart. "
CHECK_SAYS = "The cart shows one towel. "


async def check_turns(tmp: Path, kwargs: dict[str, Any], echo: str) -> dict[str, Any]:
    """T-0174 A5's two turns, with frames that make T-0207's completion check fire: every frame a real JPEG
    (``jpeg_frame``, so the fresh capture is not refused as blank), the browser's own frames ``onset, fresh, turn2``
    (the check takes the second, T-0207 ``browser_of``'s pattern) and one scripted check answer with no call."""
    from tests.fakes.llm import is_caption_prompt
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, jpeg_frame
    from tests.unit._tool_turn_support import answer, tool
    from tests.unit.test_t0117_differential import _PolicyMachine
    from tests.unit.test_t0134_label import next_turn

    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser(
        [jpeg_frame("t0211c-onset"), jpeg_frame("t0211c-fresh"), jpeg_frame("t0211c-turn2")],
        post_frames=[jpeg_frame(f"t0211c-after-{k}") for k in (1, 2, 3)],
        echo=echo,
    )
    async with feedback_rig(
        tmp,
        browser,
        script=[answer("I will add it. "), tool(a5.CLICKS[0])],
        session_kwargs=kwargs,
        machine_cls=_PolicyMachine,
    ) as rig:
        rig.llm.script_steps(
            [answer("Next. "), tool(a5.CLICKS[1])],
            [answer("Next. "), tool(a5.CLICKS[2])],
            [answer(CLAIM)],
        )
        rig.llm.script_checks([answer(CHECK_SAYS)])
        await rig.ask(a5.Q1)
        await tsl.until_steps(rig, 3, limit_ms=30_000, step=100)
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        await next_turn(rig, 2, a5.Q2, [answer("The cart says one item. ")])
        bodies = [
            body
            for (_, prompt, _), body in zip(rig.llm.generates, rig.llm.bodies, strict=True)
            if not is_caption_prompt(prompt)
        ]
        prefills = [_wire(p) for _, p in rig.llm.prefills if not is_caption_prompt(p)]
        check_bodies = list(rig.llm.check_bodies)
        log_path = Path(rig.log.path)
    records = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
    return {
        "bodies": bodies,
        "prefills": prefills,
        "check_bodies": check_bodies,
        "requests": [r["data"] for r in records if r["kind"] == "llm.request"],
    }


def test_completion_check_request_carries_v4(tmp_path: Path) -> None:
    """B-R8 (row 3): a cli-built live web-mode v4 session (live defaults: the wait tool and the completion check on)
    whose first turn ends with a no-call answer after three executed clicks sends T-0207's completion check. Its
    ``llm.request`` record (phase ``completion_check``) exists, once; its wire body is among the bodies the prefix
    property is checked over, and its rendered system turn is byte-identical to every other request's: ``<|think|>``
    first, v4's two sentences once each, before the wait sentence. (T-0174 A5's rig never fires the check: its fresh
    frame is not a JPEG, so the check has no frame and sends nothing.)"""
    v4 = v4_text()
    tpl = a5._oracle()
    kwargs, echo = a5.cli_kwargs(tmp_path / "cli", V4)
    assert kwargs.get("completion_check") is True, (
        "premise: the completion check is on by default in live (T-0207 C)"
    )
    got = asyncio.run(check_turns(tmp_path / "rig", kwargs, echo))
    phases = [r.get("phase") for r in got["requests"]]
    assert phases.count("completion_check") == 1, (
        f"T-0211 B-R8: the rig logs one completion_check request; {phases}"
    )
    assert phases.count("tool_step") == 3 and phases.count("generate") == 2, (
        f"premise: two step-1 requests and three tool steps logged; {phases}"
    )
    assert len(got["check_bodies"]) == 1 and got["check_bodies"][0], (
        "premise: one check body on the wire"
    )
    check_body = got["check_bodies"][0]
    assert any(b == check_body for b in got["bodies"]), (
        "premise: the check body is among the bodies the prefix property is checked over"
    )
    turn = a5.check_prefix_property(tpl, got, with_wait(v4))
    check_turn = a5.system_turn(a5.render(tpl, check_body))
    assert check_turn == turn, (
        "T-0211 B-R8: the check's rendered system turn is the others' byte for byte"
    )
    assert check_turn.startswith(a5.SYSTEM_OPEN), check_turn[:60]
    assert check_turn.count(S_SHARED) == 1 and check_turn.count(S_SOURCE) == 1, (
        "T-0211 B-R8: each v4 sentence is in the check's system turn once"
    )
    assert S_SOURCE + " " + WAIT in check_turn, (
        "T-0211 B-R8: the v4 sentences go before the wait sentence in the check's system turn"
    )
    check_req = got["requests"][phases.index("completion_check")]
    assert check_req["prompt"]["thinking"] is True
    logged = check_req["prompt"]["messages"][0]["content"][0]["text"]
    assert logged.endswith(S_SOURCE + " " + WAIT) and logged in check_turn, (
        "T-0211 B-R8: the logged check request carries v4 then the wait sentence"
    )
