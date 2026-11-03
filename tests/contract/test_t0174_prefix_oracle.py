"""T-0174 group A, A5: the prefix property of the live guidance, through the pinned Gemma 4 template
(docs/tasks/T-0174.md).

**What is checked.** For each source (``default``: no flag; ``v1`` and ``v2``: ``--live-guidance <v>``), ``cli``
builds a live session over the fakes (T-0152's ``_cli_run``, a real start stopped before serving) and the keywords
it passed to ``Session`` build a rig session (``feedback_rig``, the fake LLM and a fake live browser, T-0152's
reading Rd-1). The rig runs **two turns**: the first with three executed tool steps (three ``click_at`` calls, then a
spoken answer at step 4), the second a plain answer. Then:

- every wire body the fake LLM was handed (each ``generate``: step 1, steps 2-4, the second turn; built by the real
  ``VllmClient.request_body``) and every prefill prompt, rendered by the pinned template through
  ``tests/cd/oracle/gemma4_template.py`` the way vLLM renders it (image parts become ``{"type": "image"}``), has a
  system turn (up to the first ``<turn|>``) byte-identical to every other's, starting
  ``<bos><|turn>system\\n<|think|>``;
- every ``llm.request`` record of the session log carries the same system text and ``thinking: true``;
- ``chat_template_kwargs`` is identical in every body, and so is every other top-level key naming thinking;
- the system text ends with ``LIVE_GUIDANCE[v]`` (for ``default``, ``LIVE_GUIDANCE[DEFAULT_LIVE_GUIDANCE]``: v2
  since T-0174 D1, D4c re-scope; ``LIVE_TOOL_GUIDANCE`` before it);
- the v1 and v2 system turns differ only inside the guidance span.

**Adversarial** (active, the planner's named failure mode: the guidance chosen per request): the same rig with the
system message of every other request swapped for a stand-in second version (v1 with its first sentence replaced;
v2 itself does not exist at BASE) must be refused by the same checker, with the swap reaching the bodies (premise).
D4c: the swap replaces the default's guidance (v2 since D1), and the stand-in is checked to be a third text, neither
v1 nor v2, so the rig still puts two distinct system texts on the wire.

The oracle needs jinja2 (the ``cd`` dependency group) and the pinned template in the local HF cache; without them
the oracle rows skip naming what is missing. Run with::

    uv run --all-extras --group cd pytest tests/contract/test_t0174_prefix_oracle.py -q -rxX

``test_prefix_property[default]`` and ``test_alternating_versions_caught`` were green at BASE and stay green. The
``v1``/``v2`` source rows and ``test_v1_v2_differ_only_in_guidance`` were written ahead as pending (at BASE ``cli``
refused ``--live-guidance``: exit 2, a named ``AssertionError``) and activated for group A (87172bda).

Regime: CPU over fakes, ``ScriptedClock``, thinking on, tools live, loop on (8 steps), 1280x800 frames, the template
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
from tests.unit.test_session_live import _wire

FLAG = "--live-guidance"
SOURCES: dict[str, list[str]] = {"default": [], "v1": [FLAG, "v1"], "v2": [FLAG, "v2"]}
CLICKS = (
    "<|tool_call>call:click_at{x:500,y:250}<tool_call|>",
    "<|tool_call>call:click_at{x:120,y:640}<tool_call|>",
    "<|tool_call>call:click_at{x:900,y:40}<tool_call|>",
)
Q1 = "add the towel to the cart"
Q2 = "what does the cart say now"
SYSTEM_OPEN = "<bos><|turn>system\n<|think|>"
#: the adversarial stand-in second version: v1 with only its first sentence replaced (A2's shape)
ALT = (
    "The screenshot attached here is the screen after your last action. "
    + pb.LIVE_TOOL_GUIDANCE.split(". ", 1)[1]
)


def _oracle() -> Any:
    pytest.importorskip("jinja2", reason="the template oracle needs jinja2: uv run --group cd ...")
    from tests.cd.oracle import gemma4_template as tpl

    if not tpl.template_path().is_file():
        pytest.skip(f"the pinned chat template is not in the local HF cache: {tpl.template_path()}")
    return tpl


def guidance_of(source: str) -> str:
    """The guidance text a source's session carries: ``LIVE_GUIDANCE[v]`` for ``v1``/``v2``; for ``default`` (no
    flag) ``LIVE_GUIDANCE[config.DEFAULT_LIVE_GUIDANCE]`` (T-0174 D4c: v2 since D1, v1 before)."""
    m = getattr(pb, "LIVE_GUIDANCE", None)
    assert m is not None, "T-0174 A2: prompt_builder.LIVE_GUIDANCE does not exist yet"
    if source == "default":
        source = config.DEFAULT_LIVE_GUIDANCE
    # T-0202 D3 re-pin (admit, the default): with the wait tool on by default in live, every source's live session
    # (none names --wait-tool) carries the version's text, one space, then H-T202-4's sentence (typed in
    # _t0202_pins from the task), so the prefix property is checked on the session cli really builds; on a tree with
    # the default off this is the version's text alone, as before
    from tests.unit import _t0202_pins as t0202

    text = str(m[source])
    return text + " " + t0202.SENTENCE if t0202.default_on() else text


# --------------------------------------------------------------------------------------------- the rig


def cli_kwargs(tmp: Path, flags: list[str]) -> tuple[dict[str, Any], str]:
    """The ``Session`` keywords and executor echo of a live ``cli`` start over the fakes (T-0152's ``_cli_run``)."""
    from tests.unit import test_t0152_overview_frame_default as t0152

    run = t0152._cli_run(tmp, flags, "", dry=False)
    assert run["code"] == 0 and run["_raw_kwargs"], (
        f"T-0174 A4: astra session --tools live {' '.join(flags)} builds a session; exit {run['code']}\n"
        f"{run['out'][-600:]}{run['err'][-600:]}"
    )
    return t0152.rig_kwargs(run)


def swap_system(prompt: Prompt, guidance: str, current: str | None = None) -> Prompt:
    """``prompt`` with its system message's guidance (``current``, default: the no-flag session's,
    ``guidance_of("default")``) replaced by ``guidance`` (the adversarial mutation)."""
    current = guidance_of("default") if current is None else current
    first = prompt.messages[0]
    assert first.role == "system"
    text = first.text()
    assert text.endswith(current), "premise: the rig's system text ends with the default's guidance"
    swapped = Message(
        role="system",
        content=(TextPart(text=text[: -len(current)] + guidance),),
    )
    return prompt.model_copy(update={"messages": (swapped, *prompt.messages[1:])})


async def two_turns(
    tmp: Path, kwargs: dict[str, Any], echo: str, *, alternate: bool
) -> dict[str, Any]:
    """Turn 1: three executed clicks then an answer; turn 2: an answer. Returns the wire bodies of every generate
    (captioner prompts excluded), the prefill prompts' bodies, and the session log's ``llm.request`` records."""
    from tests.fakes.llm import is_caption_prompt
    from tests.unit import test_session_tool_loop as tsl
    from tests.unit._live_browser_support import FakeLiveBrowser, feedback_rig, live_frame
    from tests.unit._tool_turn_support import answer, tool
    from tests.unit.test_t0117_differential import _PolicyMachine
    from tests.unit.test_t0134_label import next_turn

    tmp.mkdir(parents=True, exist_ok=True)
    browser = FakeLiveBrowser(
        [live_frame("t0174-onset"), live_frame("t0174-turn2")],
        post_frames=[live_frame(f"t0174-after-{k}") for k in (1, 2, 3)],
        echo=echo,
    )
    async with feedback_rig(
        tmp,
        browser,
        script=[answer("I will add it. "), tool(CLICKS[0])],
        session_kwargs=kwargs,
        machine_cls=_PolicyMachine,
    ) as rig:
        if alternate:
            real = rig.llm.generate
            count = [0]

            def generate(prompt: Prompt, *a: Any, **k: Any) -> Any:
                count[0] += 1
                if count[0] % 2 == 0:
                    prompt = swap_system(prompt, ALT)
                return real(prompt, *a, **k)

            rig.llm.generate = generate
        rig.llm.script_steps(
            [answer("Next. "), tool(CLICKS[1])],
            [answer("Next. "), tool(CLICKS[2])],
            [answer("The towel is in the cart. ")],
        )
        await rig.ask(Q1)
        await tsl.until_steps(rig, 3, limit_ms=30_000, step=100)
        await rig.turn_done(limit_ms=30_000)
        await rig.advance(300)
        await next_turn(rig, 2, Q2, [answer("The cart says one item. ")])
        executed = sum(1 for c in browser.calls if c[0] == "execute" and "-call" in str(c[1]))
        bodies = [
            body
            for (_, prompt, _), body in zip(rig.llm.generates, rig.llm.bodies, strict=True)
            if not is_caption_prompt(prompt)
        ]
        prefills = [_wire(p) for _, p in rig.llm.prefills if not is_caption_prompt(p)]
        lines = rig.lines()
        session_id = rig.session.session_id
        log_path = Path(rig.log.path)
    records = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
    return {
        "bodies": bodies,
        "prefills": prefills,
        "requests": [r["data"] for r in records if r["kind"] == "llm.request"],
        "executed": executed,
        "turns": len(lines),
        "session_id": session_id,
    }


def run_rig(tmp: Path, flags: list[str], *, alternate: bool = False) -> dict[str, Any]:
    kwargs, echo = cli_kwargs(tmp / "cli", flags)
    got = asyncio.run(two_turns(tmp / "rig", kwargs, echo, alternate=alternate))
    assert got["turns"] == 2 and got["executed"] == 3, (
        f"premise: two turns, three executed clicks; {got['turns']} turns, {got['executed']} executed"
    )
    assert len(got["bodies"]) >= 5, f"premise: 4 + 1 generates; {len(got['bodies'])}"
    phases = [r.get("phase") for r in got["requests"]]
    assert phases.count("tool_step") == 3 and phases.count("generate") == 2, (
        f"premise: two step-1 requests and three tool steps logged; {phases}"
    )
    return got


# --------------------------------------------------------------------------------------------- rendering


def _oracle_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    return [{"type": "image"} if p.get("type") == "image_url" else dict(p) for p in content]


def render(tpl: Any, body: dict[str, Any]) -> str:
    messages = [
        {"role": m["role"], "content": _oracle_content(m["content"])} for m in body["messages"]
    ]
    cont = bool(body.get("continue_final_message", False))
    text: str = tpl.render(
        messages,
        tools=body.get("tools"),
        add_generation_prompt=not cont,
        enable_thinking=(body.get("chat_template_kwargs") or {}).get("enable_thinking"),
    )
    return text


def system_turn(text: str) -> str:
    end = text.index("<turn|>")
    return text[: end + len("<turn|>")]


def _thinking_keys(body: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in body.items() if k == "chat_template_kwargs" or "think" in k.lower()}


def check_prefix_property(tpl: Any, got: dict[str, Any], guidance: str) -> str:
    """A5's checks (module docstring); returns the one rendered system turn."""
    rendered = [system_turn(render(tpl, b)) for b in [*got["prefills"], *got["bodies"]]]
    distinct = sorted(set(rendered))
    assert len(distinct) == 1, (
        f"T-0174 A5: the rendered system turn varies across {len(rendered)} requests ({len(distinct)} distinct); "
        f"first difference at "
        f"{next(i for i, (x, y) in enumerate(zip(distinct[0], distinct[1], strict=False)) if x != y)}"
    )
    turn = distinct[0]
    assert turn.startswith(SYSTEM_OPEN), (
        f"T-0174 A5: <|think|> is the first system token: {turn[:60]!r}"
    )
    kw = {json.dumps(_thinking_keys(b), sort_keys=True) for b in got["bodies"]}
    assert len(kw) == 1, (
        f"T-0174 A5: chat_template_kwargs / thinking flags differ between requests: {kw}"
    )
    assert json.loads(next(iter(kw))).get("chat_template_kwargs", {}).get("enable_thinking") is True
    texts = set()
    for r in got["requests"]:
        msgs = r["prompt"]["messages"]
        assert msgs[0]["role"] == "system", msgs[0]["role"]
        texts.add(msgs[0]["content"][0]["text"])
        assert r["prompt"]["thinking"] is True, r.get("phase")
    assert len(texts) == 1, f"T-0174 A5: the logged system text varies ({len(texts)} distinct)"
    text = next(iter(texts))
    assert text.endswith("\n" + guidance) and text.count(guidance) == 1, (
        f"T-0174 A5: the system text ends with the version's guidance; its tail {text[-200:]!r}"
    )
    assert text in turn, "premise: the logged system text is what the template rendered"
    return turn


# --------------------------------------------------------------------------------------------- rows


@pytest.mark.parametrize("source", list(SOURCES))
def test_prefix_property(tmp_path: Path, source: str) -> None:
    """A5: one system turn across every request of two turns (the first with three tool steps), ``<|think|>``
    first, the same thinking flags, ending with the version's guidance."""
    guidance = guidance_of(source)
    tpl = _oracle()
    got = run_rig(tmp_path, SOURCES[source])
    check_prefix_property(tpl, got, guidance)


def test_v1_v2_differ_only_in_guidance(tmp_path: Path) -> None:
    """A5: the v1 and v2 rendered system turns are equal outside the guidance span: v2's is v1's with
    ``LIVE_TOOL_GUIDANCE`` replaced by ``LIVE_GUIDANCE["v2"]`` at the same offset, once."""
    v1, v2 = guidance_of("v1"), guidance_of("v2")
    tpl = _oracle()
    t1 = check_prefix_property(tpl, run_rig(tmp_path / "v1", SOURCES["v1"]), v1)
    t2 = check_prefix_property(tpl, run_rig(tmp_path / "v2", SOURCES["v2"]), v2)
    assert t1.count(v1) == 1, "premise: v1 appears once in its system turn"
    i = t1.index(v1)
    assert t2 == t1[:i] + v2 + t1[i + len(v1) :], (
        "T-0174 A5: the v1 and v2 system turns differ outside the guidance span"
    )
    assert t1 != t2


def test_alternating_versions_caught(tmp_path: Path) -> None:
    """A5 adversarial (active): a rig whose every other request carries a second guidance version (v1 with its first
    sentence replaced) is refused by ``check_prefix_property``; premise: the swap reached the bodies (two distinct
    system texts on the wire) and the unswapped default passes the same checker."""
    tpl = _oracle()
    default = guidance_of("default")
    assert ALT not in (default, pb.LIVE_TOOL_GUIDANCE, pb.LIVE_GUIDANCE["v2"]), (
        "premise: the stand-in is a third text, neither the default's guidance nor either version"
    )
    got = run_rig(tmp_path, SOURCES["default"], alternate=True)
    flat = {json.dumps(b["messages"][0]["content"]) for b in got["bodies"]}
    assert len(flat) == 2, f"premise: the mutation put two system texts on the wire; {len(flat)}"
    texts = [b["messages"][0]["content"] for b in got["bodies"]]
    assert any(json.dumps(default)[1:-1] in json.dumps(t) for t in texts) and any(
        json.dumps(ALT)[1:-1] in json.dumps(t) for t in texts
    ), "premise: the wire carries the default's guidance and the stand-in, alternately"
    with pytest.raises(AssertionError, match="T-0174 A5: the rendered system turn varies"):
        check_prefix_property(tpl, got, default)
