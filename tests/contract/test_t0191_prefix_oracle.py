"""T-0191 group A, A5: the prefix property of live guidance v3, through the pinned Gemma 4 template
(docs/tasks/T-0191.md).

**What is checked** (T-0174 A5's rig and checker, reused): ``cli`` builds a live session over the fakes (T-0152's
``_cli_run``: ``astra session --tools live --browser headless <flags>``, a real start stopped before serving) and the
keywords it passed to ``Session`` build a rig session over the fake LLM and a fake live browser. The rig runs two
turns, the first with three executed ``click_at`` steps then an answer, the second a plain answer. Then
``test_t0174_prefix_oracle.check_prefix_property``: every wire body and prefill, rendered by the pinned template
(``tests/cd/oracle/gemma4_template.py``), has one byte-identical system turn starting
``<bos><|turn>system\\n<|think|>``; every ``llm.request`` record carries that system text and ``thinking: true``; the
thinking keys and ``chat_template_kwargs`` are identical in every body; the system text ends with the guidance once.

- ``test_prefix_property_v3``: ``--live-guidance v3`` in web mode; the guidance is ``LIVE_GUIDANCE["v3"]``.
- ``test_v3_v2_differ_only_by_sentence``: the v3 and v2 rendered system turns are equal except for ``" " + S``
  inserted right after v2's text (the end of the guidance span), once.
- ``test_no_web_v3_is_v2s_turn``: ``--live-guidance v3 --no-web`` renders exactly the system turn of
  ``--live-guidance v2 --no-web`` (A4: the v2 object itself).
- Adversarial ``test_web_state_switch_caught`` (the planner's named failure: the text chosen per request, here per
  web state): a rig whose second-turn requests carry the ``--no-web`` text (v2's) while the first turn's carried v3's
  is refused by the same checker; premise: both texts reached the wire. It is written against v3, so it is pending
  with the rest.

The oracle needs jinja2 (the ``cd`` dependency group) and the pinned template in the local HF cache; without them the
rows skip naming what is missing. Run with::

    uv run --all-extras --group cd pytest tests/contract/test_t0191_prefix_oracle.py -q -rxX

Every row was written ahead as pending and activated at A's ``/implement`` (2026-10-28): at BASE (f1caef6a) ``cli`` refuses ``--live-guidance v3``
(argparse ``invalid choice``, exit 2) and ``LIVE_TOOL_GUIDANCE_V3`` does not exist, each a named ``AssertionError``.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, tools live, web mode unless ``--no-web``, loop on, 1280x800
frames, the template of ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev 2ea8372; n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import prompt_builder as pb
from local_astra.contracts.llm import Message, Prompt, TextPart
from tests.contract import test_t0174_prefix_oracle as a5
from tests.unit.test_session_live import _wire

FLAG = "--live-guidance"
V3 = [FLAG, "v3"]
V2 = [FLAG, "v2"]


def v3_text() -> str:
    value = getattr(pb, "LIVE_TOOL_GUIDANCE_V3", None)
    assert isinstance(value, str), (
        "T-0191 A1: prompt_builder.LIVE_TOOL_GUIDANCE_V3 does not exist yet"
    )
    return value


def sentence() -> str:
    v3, v2 = v3_text(), pb.LIVE_TOOL_GUIDANCE_V2
    assert v3.startswith(v2 + " ") and len(v3) > len(v2) + 1, (
        f"T-0191 A1: v3 is v2 + ' ' + S: {v3!r}"
    )
    return v3[len(v2) + 1 :]


def g(text: str) -> str:  # T-0202 D3 re-pin
    """T-0202 D3 re-pin (admit, the default; as ``test_t0174_prefix_oracle.guidance_of``): with the wait tool on by
    default in live, a live session that names no ``--wait-tool`` carries the version's text, one space, then
    H-T202-4's sentence (typed in ``_t0202_pins`` from the task); on a tree with the default off, the text alone."""
    from tests.unit import _t0202_pins as t0202

    return text + " " + t0202.SENTENCE if t0202.default_on() else text


def run_rig(tmp: Path, flags: list[str]) -> dict[str, Any]:
    """T-0174 A5's ``run_rig`` (its premises: two turns, three executed clicks, two generate and three tool-step
    requests logged)."""
    return a5.run_rig(tmp, flags)


def swap_guidance(prompt: Prompt, current: str, replacement: str) -> Prompt:
    first = prompt.messages[0]
    assert first.role == "system"
    text = first.text()
    assert text.endswith(current), "premise: the rig's system text ends with the session's guidance"
    swapped = Message(role="system", content=(TextPart(text=text[: -len(current)] + replacement),))
    return prompt.model_copy(update={"messages": (swapped, *prompt.messages[1:])})


async def switching_turns(tmp: Path, kwargs: dict[str, Any], echo: str) -> dict[str, Any]:
    """T-0174 A5's two turns, with every request of the second turn carrying v2's text in place of v3's (the web state
    switched between turns)."""
    from tests.fakes.llm import is_caption_prompt
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_frame
    from tests.unit._tool_turn_support import answer, tool
    from tests.unit.test_t0117_differential import _PolicyMachine
    from tests.unit.test_t0134_label import next_turn

    v3, v2 = v3_text(), pb.LIVE_TOOL_GUIDANCE_V2
    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser(
        [live_frame("t0191-onset"), live_frame("t0191-turn2")],
        post_frames=[live_frame(f"t0191-after-{k}") for k in (1, 2, 3)],
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
                prompt = swap_guidance(
                    prompt, g(v3), g(v2)
                )  # T-0202 D3 re-pin: the session's whole guidance
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


def test_prefix_property_v3(tmp_path: Path) -> None:
    """A5: a cli-built live web-mode v3 session: one rendered system turn across every request of two turns (the
    first with three tool steps), ``<|think|>`` first, the same thinking flags, ending with v3's guidance once."""
    v3 = v3_text()
    assert pb.LIVE_GUIDANCE.get("v3") is v3, (
        "T-0191 A3: LIVE_GUIDANCE['v3'] is LIVE_TOOL_GUIDANCE_V3"
    )
    tpl = a5._oracle()
    a5.check_prefix_property(tpl, run_rig(tmp_path, V3), g(v3))  # T-0202 D3 re-pin


def test_v3_v2_differ_only_by_sentence(tmp_path: Path) -> None:
    """A5: the v3 and v2 rendered system turns (web mode) differ only by ``" " + S`` at the end of the guidance span:
    v3's turn is v2's with ``" " + S`` inserted right after v2's guidance text, and v2's text occurs once."""
    v2, v3, s = pb.LIVE_TOOL_GUIDANCE_V2, v3_text(), sentence()
    tpl = a5._oracle()
    t2 = a5.check_prefix_property(tpl, run_rig(tmp_path / "v2", V2), g(v2))  # T-0202 D3 re-pin
    t3 = a5.check_prefix_property(tpl, run_rig(tmp_path / "v3", V3), g(v3))  # T-0202 D3 re-pin
    assert t2.count(v2) == 1, "premise: v2 appears once in its system turn"
    i = t2.index(v2) + len(v2)
    assert t3 == t2[:i] + " " + s + t2[i:], (
        "T-0191 A5: the v3 and v2 system turns differ other than by ' ' + S after the guidance"
    )


def test_no_web_v3_is_v2s_turn(tmp_path: Path) -> None:
    """A4/A5: ``--live-guidance v3 --no-web`` renders the same system turn as ``--live-guidance v2 --no-web`` (and
    holds the prefix property with v2's text); S is not in it."""
    v2, s = pb.LIVE_TOOL_GUIDANCE_V2, sentence()
    tpl = a5._oracle()
    # T-0202 D3 re-pin: the guidance each session carries (g)
    t2 = a5.check_prefix_property(tpl, run_rig(tmp_path / "v2", [*V2, "--no-web"]), g(v2))
    t3 = a5.check_prefix_property(tpl, run_rig(tmp_path / "v3", [*V3, "--no-web"]), g(v2))
    assert t3 == t2 and s not in t3, "T-0191 A4: v3 without web mode is not v2's system turn"


def test_web_state_switch_caught(tmp_path: Path) -> None:
    """A5 adversarial: a rig whose second turn carries the ``--no-web`` text (v2) after a first turn with v3's is
    refused by ``check_prefix_property``; premise: both texts reached the wire, and the unswitched v3 rig passes the
    same checker."""
    v2, v3 = pb.LIVE_TOOL_GUIDANCE_V2, v3_text()
    tpl = a5._oracle()
    a5.check_prefix_property(tpl, run_rig(tmp_path / "plain", V3), g(v3))  # T-0202 D3 re-pin
    kwargs, echo = a5.cli_kwargs(tmp_path / "cli", V3)
    got = asyncio.run(switching_turns(tmp_path / "rig", kwargs, echo))
    flat = {json.dumps(b["messages"][0]["content"]) for b in got["bodies"]}
    assert len(flat) == 2, f"premise: the switch put two system texts on the wire; {len(flat)}"
    texts = [json.dumps(b["messages"][0]["content"]) for b in got["bodies"]]
    assert any(json.dumps(v3)[1:-1] in t for t in texts), "premise: v3 on the wire"
    assert any(json.dumps(v3)[1:-1] not in t and json.dumps(v2)[1:-1] in t for t in texts), (
        "premise: v2 alone on the wire"
    )
    with pytest.raises(AssertionError, match="T-0174 A5: the rendered system turn varies"):
        a5.check_prefix_property(tpl, got, g(v3))  # T-0202 D3 re-pin
