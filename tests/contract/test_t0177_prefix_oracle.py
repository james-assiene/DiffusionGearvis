"""T-0177 A9's last sentence and the implementation note "Prefix cache": the system message bytes are identical
across browser and shared turns, ``<|think|>`` first, rendered through the pinned Gemma 4 template (T-0174 A5's
pattern and checker pieces, imported from ``test_t0174_prefix_oracle``).

**The rig.** ``test_t0177_shared_frame.vision_rig`` in live (the fake live browser, the HUD perception holding the
distinctive shared frame ``SH``, a socketless ``HudServer`` with one greeted page). Three turns, **browser, shared,
browser**, each with two steps: step 1 a spoken sentence and a ``scroll`` (a non-coordinate action, executed under
both sources in groups A and B), step 2 a spoken answer. The choice is changed between turns by
``astra.vision.select`` from the page.

**The checker** (``check_one_system_turn``): every prefill prompt and every generate body of the session (captions
excluded), rendered by the pinned template the way vLLM renders it, has one system turn (up to the first ``<turn|>``),
byte-identical across all of them and starting ``<bos><|turn>system\\n<|think|>``; the thinking keys
(``chat_template_kwargs`` and any top-level key naming thinking) are identical in every body with
``enable_thinking: true``; every ``llm.request`` record carries one system text and ``thinking: true``, and that text
is inside the rendered system turn.

**Rows.**

- ``test_one_system_turn_across_sources`` (red until A3/A6/A7 land):the checker passes on the three-turn rig; premises: turn 2 really
  is shared (its step-1 body carries ``SH``'s bytes and ``SHARED_LABEL_SCREEN``, the label in the final user message,
  never in the system message), turns 1 and 3 carry no shared bytes, and 3 x 2 generates were sent.
- ``test_a_source_sentence_in_the_system_message_is_caught`` (active, green at BASE): the same rig with a source
  sentence inserted into the system message of **every turn-2 request** (prefill and generates; the planner's named
  failure mode, "a source sentence in the system prompt") must be refused by the same checker, with the mutation
  reaching the wire (premise: two distinct system texts in the bodies). Its selects are raw JSON (no asserting
  helper), so it runs at BASE, where they are counted ``unhandled`` and every turn is a browser turn.

The oracle needs jinja2 (the ``cd`` dependency group) and the pinned template in the local HF cache; without them the
rows skip naming what is missing. Run with::

    uv run --all-extras --group cd pytest tests/contract/test_t0177_prefix_oracle.py -q -rxX

Regime: CPU over fakes, ``ScriptedClock``, thinking on, tools live, loop on (8 steps), 1280x800 browser frames, the
1568x590 shared frame, the template of ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev 2ea8372; n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.llm import Message, Prompt, TextPart
from tests.contract.test_t0174_prefix_oracle import (
    SYSTEM_OPEN,
    _oracle,
    _thinking_keys,
    render,
    system_turn,
)
from tests.unit._live_browser_support import b64, live_frame
from tests.unit._tool_turn_support import answer, tool
from tests.unit.test_session_live import SCROLL
from tests.unit.test_t0177_shared_frame import (
    SELECT,
    SH,
    SHARED_LABEL_SCREEN,
    VisionBrowser,
    select,
    vision_rig,
)

SOURCES = ("browser", "shared", "browser")
QUESTIONS = ("scroll the shop page", "what is on my screen", "scroll it again")
#: the adversarial source sentence (what a per-source system prompt would add)
SOURCE_SENTENCE = "You are looking at the user's shared screen, not your browser."


def insert_sentence(prompt: Prompt) -> Prompt:
    """``prompt`` with ``SOURCE_SENTENCE`` appended to its system message (the adversarial mutation)."""
    first = prompt.messages[0]
    assert first.role == "system", "premise: the first message is the system message"
    swapped = Message(
        role="system", content=(TextPart(text=first.text() + "\n" + SOURCE_SENTENCE),)
    )
    return prompt.model_copy(update={"messages": (swapped, *prompt.messages[1:])})


def _is_turn2(request_id: str) -> bool:
    return "-t2-" in request_id or request_id.startswith("t2-")


async def three_turns(tmp: Path, *, strict_select: bool, mutate: bool) -> dict[str, Any]:
    """Browser, shared, browser; two steps each. Returns the generate bodies (captions excluded, each with its
    request id), the prefill prompts' bodies, and the session log's ``llm.request`` records."""
    from tests.fakes.llm import is_caption_prompt
    from tests.unit.test_session_live import _wire

    tmp.mkdir(parents=True, exist_ok=True)
    browser = VisionBrowser(
        [live_frame(f"t0177-px-onset-{k}") for k in (1, 2, 3)],
        post_frames=[live_frame(f"t0177-px-after-{k}") for k in (1, 2, 3)],
    )
    async with vision_rig(tmp, browser, script=(), hud_frames=[SH]) as rig:
        if mutate:
            real_generate, real_prefill = rig.llm.generate, rig.llm.prefill

            def generate(prompt: Prompt, request_id: str, *a: Any, **k: Any) -> Any:
                if _is_turn2(request_id):
                    prompt = insert_sentence(prompt)
                return real_generate(prompt, request_id, *a, **k)

            async def prefill(prefix: Prompt, request_id: str) -> Any:
                if _is_turn2(request_id):
                    prefix = insert_sentence(prefix)
                return await real_prefill(prefix, request_id)

            rig.llm.generate, rig.llm.prefill = generate, prefill
        current = "browser"
        for n, (source, question) in enumerate(zip(SOURCES, QUESTIONS, strict=True), start=1):
            if source != current:
                if strict_select:
                    await select(rig, source)
                else:
                    rig.page.receive(json.dumps({"type": SELECT, "source": source}))
                    await rig.pump()
                current = source
            rig.llm.script = [answer("Let me look. "), tool(SCROLL)]
            rig.llm.script_steps([answer(f"Done with turn {n}. ")])
            await rig.ask(question)
            await rig.until(
                lambda n=n: rig.state.state.name == "IDLE" and len(rig.lines()) >= n,
                f"turn {n} never ended",
                limit_ms=30_000,
                step=100,
            )
            await rig.advance(300)
        bodies = [
            {"request_id": rid, "body": body}
            for (rid, prompt, _), body in zip(rig.llm.generates, rig.llm.bodies, strict=True)
            if not is_caption_prompt(prompt)
        ]
        prefills = [_wire(p) for _, p in rig.llm.prefills if not is_caption_prompt(p)]
        lines = rig.lines()
        log_path = Path(rig.log.path)
        executed = sum(1 for i in browser.invocations if i.name == "scroll")
    records = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
    return {
        "bodies": bodies,
        "prefills": prefills,
        "requests": [r["data"] for r in records if r["kind"] == "llm.request"],
        "turns": len(lines),
        "executed": executed,
    }


def run_rig(tmp: Path, *, strict_select: bool, mutate: bool = False) -> dict[str, Any]:
    got = asyncio.run(three_turns(tmp, strict_select=strict_select, mutate=mutate))
    assert got["turns"] == 3 and got["executed"] == 3, (
        f"premise: three turns, three executed scrolls; {got['turns']} turns, {got['executed']} executed"
    )
    phases = [r.get("phase") for r in got["requests"]]
    assert phases.count("generate") == 3 and phases.count("tool_step") == 3, (
        f"premise: three step-1 requests and three tool steps logged; {phases}"
    )
    return got


def check_one_system_turn(tpl: Any, got: dict[str, Any]) -> str:
    """The checker (module docstring); returns the one rendered system turn."""
    bodies = [b["body"] for b in got["bodies"]]
    rendered = [system_turn(render(tpl, b)) for b in [*got["prefills"], *bodies]]
    distinct = sorted(set(rendered))
    assert len(distinct) == 1, (
        f"T-0177 A9: the rendered system turn varies across {len(rendered)} requests ({len(distinct)} distinct); "
        f"first difference at "
        f"{next(i for i, (x, y) in enumerate(zip(distinct[0], distinct[1], strict=False)) if x != y)}"
    )
    turn = distinct[0]
    assert turn.startswith(SYSTEM_OPEN), (
        f"T-0177 A9: <|think|> is the first system token: {turn[:60]!r}"
    )
    kw = {json.dumps(_thinking_keys(b), sort_keys=True) for b in bodies}
    assert len(kw) == 1, (
        f"T-0177 A9: chat_template_kwargs / thinking flags differ between requests: {kw}"
    )
    assert json.loads(next(iter(kw))).get("chat_template_kwargs", {}).get("enable_thinking") is True
    texts = set()
    for r in got["requests"]:
        msgs = r["prompt"]["messages"]
        assert msgs[0]["role"] == "system", msgs[0]["role"]
        texts.add(msgs[0]["content"][0]["text"])
        assert r["prompt"]["thinking"] is True, r.get("phase")
    assert len(texts) == 1, f"T-0177 A9: the logged system text varies ({len(texts)} distinct)"
    assert next(iter(texts)) in turn, (
        "premise: the logged system text is what the template rendered"
    )
    return turn


def _final_user(body: dict[str, Any]) -> dict[str, Any]:
    last: dict[str, Any] = [m for m in body["messages"] if m["role"] == "user"][-1]
    return last


def _final_user_texts(body: dict[str, Any]) -> list[str]:
    content = _final_user(body)["content"]
    if isinstance(content, str):
        return [content]
    return [str(p.get("text", "")) for p in content if p.get("type") == "text"]


def test_one_system_turn_across_sources(tmp_path: Path) -> None:
    """A9 / "Prefix cache": one rendered system turn, ``<|think|>`` first, across every request of a browser, a
    shared and a browser turn (two steps each); the shared label lives in the final user message only."""
    tpl = _oracle()
    got = run_rig(tmp_path, strict_select=True)
    shared_b64 = b64(SH.jpeg)
    by_turn: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for b in got["bodies"]:
        for n in by_turn:
            if f"-t{n}-" in b["request_id"] or b["request_id"].startswith(f"t{n}-"):
                by_turn[n].append(b["body"])
    assert [len(v) for v in by_turn.values()] == [2, 2, 2], (
        f"premise: two generates per turn; {[b['request_id'] for b in got['bodies']]}"
    )
    step1_t2 = by_turn[2][0]
    assert shared_b64 in json.dumps(step1_t2), (
        "premise (A6): turn 2's step 1 carries the shared frame"
    )
    assert SHARED_LABEL_SCREEN in _final_user_texts(step1_t2), (
        "premise (A7): the shared label is in turn 2's final user message"
    )
    # Turn 1 precedes every select, so no part of its bodies carries the shared frame (B6 under browser).
    assert not [b for b in by_turn[1] if shared_b64 in json.dumps(b)], (
        "premise: browser turn 1 carries no shared bytes"
    )
    # Turn 3 is a browser turn: its current image (the final user message) is never the shared frame. Turn 2's
    # shared frame stays in turn 3's *history* (BASE keeps images for two turns; H-T177-4 keeps history images of
    # the other source), so the whole-body search of the first version of this premise was wrong.
    for i, b in enumerate(by_turn[3], start=1):
        assert shared_b64 not in json.dumps(_final_user(b)), (
            f"premise: browser turn 3 step {i}'s final user message carries no shared bytes"
        )
        history = [m for m in b["messages"][1:] if m is not _final_user(b)]
        assert shared_b64 in json.dumps(history), (
            f"H-T177-4: turn 2's shared image is still in turn 3 step {i}'s history"
        )
    for b in [*got["prefills"], *(x["body"] for x in got["bodies"])]:
        system = json.dumps(b["messages"][0])
        assert "shared screen" not in system and SHARED_LABEL_SCREEN not in system, (
            "T-0177 A7 / prefix cache: no source wording in the system message"
        )
    check_one_system_turn(tpl, got)


def test_a_source_sentence_in_the_system_message_is_caught(tmp_path: Path) -> None:
    """Adversarial (active): a source sentence inserted into the system message of every turn-2 request (prefill and
    generates) is refused by ``check_one_system_turn``. Premises: the mutation reached the wire (two distinct system
    texts across the bodies, the mutated one only on turn 2), and the unmutated rig passes the same checker."""
    tpl = _oracle()
    clean = run_rig(tmp_path / "clean", strict_select=False)
    check_one_system_turn(tpl, clean)
    got = run_rig(tmp_path / "mutated", strict_select=False, mutate=True)
    systems = {
        b["request_id"]: json.dumps(b["body"]["messages"][0]["content"]) for b in got["bodies"]
    }
    assert len(set(systems.values())) == 2, (
        f"premise: the mutation put two system texts on the wire; {len(set(systems.values()))}"
    )
    marked = sorted(rid for rid, s in systems.items() if SOURCE_SENTENCE in s)
    assert marked and all(_is_turn2(rid) for rid in marked) and len(marked) == 2, (
        f"premise: exactly turn 2's two generates carry the sentence; {marked}"
    )
    with pytest.raises(AssertionError, match="T-0177 A9: the rendered system turn varies"):
        check_one_system_turn(tpl, got)
