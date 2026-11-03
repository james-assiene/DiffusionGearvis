"""T-0131 C1-C3: the two live-only guidance sentences in the system message (docs/tasks/T-0131.md, group C).

The interfaces the rows pin (the task's): ``prompt_builder.LIVE_TOOL_GUIDANCE: str`` (a module-level literal) and
``PromptBuilder(system_prompt, budget=None, thinking=True, *, tools=None, tool_guidance: str | None = None)``. With
``tool_guidance`` set **and** tools on, the text is appended after ``NATIVE_TOOL_INSTRUCTIONS``, last in the system
message; otherwise the system message is BASE's. Which sessions get it (``cli``, ``tools = live`` only) is
``test_cli_t0131.py::test_live_gets_guidance_dry_run_does_not``.

Oracles: the tools-on system text without guidance is BASE's (pinned by ``test_session_prompt.py``); the guided
text must equal it plus a whitespace-only separator plus ``LIVE_TOOL_GUIDANCE``, which is the criterion's
"appended ... last". The wording is the implementer's; ``test_mentions_check_and_scroll`` and
``test_no_confirmation_or_narration_request`` check keywords only, and the reviewer judges the meaning (C1's
(a) and (b)).

Every row is ``pending(task="T-0131")``: at BASE (f30365d) neither name exists, so each fails on a named
``AssertionError``.

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
from typing import Any

from local_astra.conductor import prompt_builder as pb
from local_astra.conductor.prompt_builder import PromptBuilder
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.turn import HistoryTurn
from tests.unit import _tools_switch_support as T

SYSTEM = "You are Astra, a helpful assistant."


def guidance() -> str:
    value = getattr(pb, "LIVE_TOOL_GUIDANCE", None)
    assert isinstance(value, str), (
        "T-0131 C1: prompt_builder.LIVE_TOOL_GUIDANCE (a str) does not exist yet"
    )
    return value


def builder(*, tools: Any = None, tool_guidance: str | None = None, **kw: Any) -> PromptBuilder:
    params = inspect.signature(PromptBuilder.__init__).parameters
    assert "tool_guidance" in params, (
        f"T-0131 C1: PromptBuilder(..., tool_guidance: str | None = None) does not exist yet; it takes {sorted(params)}"
    )
    return PromptBuilder(SYSTEM, tools=tools, tool_guidance=tool_guidance, **kw)


def system_text(b: PromptBuilder) -> str:
    return b.system_message().text()


def _frame(tag: str) -> Frame:
    return Frame(
        source=Source.WINDOW,
        ts_ms=0,
        width=1280,
        height=800,
        jpeg=b"\xff\xd8" + tag.encode() * 8,
        phash=hashlib.sha256(tag.encode()).hexdigest()[:16],
    )


def test_appended_last_live_only() -> None:
    """C1 (mutation 3's prompt half): with tools on and ``tool_guidance`` set, the system text is the tools-on text,
    a whitespace-only separator, then ``LIVE_TOOL_GUIDANCE``, last and once; with tools off the guidance is ignored
    (the text is BASE's tools-off text); with ``tool_guidance=None`` the tools-on text is BASE's. Every prompt the
    builder returns (``prefix`` and ``full``) carries that system message first."""
    g = guidance()
    tools = T.tools_field()
    on = system_text(PromptBuilder(SYSTEM, tools=tools))
    off = system_text(PromptBuilder(SYSTEM))
    guided = system_text(builder(tools=tools, tool_guidance=g))
    assert guided.startswith(on), "the guided text begins with the tools-on text, byte for byte"
    assert guided.endswith(g), "T-0131 C1: LIVE_TOOL_GUIDANCE is last in the system message"
    sep = guided[len(on) : len(guided) - len(g)]
    assert sep and sep.strip() == "", (
        f"only a separator between NATIVE_TOOL_INSTRUCTIONS and the guidance: {sep!r}"
    )
    assert guided.count(g) == 1
    assert on.endswith(pb.NATIVE_TOOL_INSTRUCTIONS), (
        "premise: the tools-on text ends with the native instructions"
    )
    assert system_text(builder(tools=None, tool_guidance=g)) == off, (
        "T-0131 C1: with tools off the guidance is never added"
    )
    assert g not in off and g not in on
    assert system_text(builder(tools=tools, tool_guidance=None)) == on
    b = builder(tools=tools, tool_guidance=g)
    prefix = b.prefix([], frames=(_frame("a"),))
    full = b.full(prefix, "add the blue mug to the cart", None, [])
    for prompt in (prefix, full):
        assert prompt.messages[0].role == "system" and prompt.messages[0].text() == guided
        assert sum(m.text().count(g) for m in prompt.messages) == 1, (
            "the guidance appears in the system message only"
        )


def test_literal_identical_across_turns_and_thinking() -> None:
    """C2: ``LIVE_TOOL_GUIDANCE`` is a literal (a module-level ``str`` constant, not assembled at run time), and the
    system message carrying it is byte-identical with thinking on and off, across turns (with and without history
    and frames) and across builders (so across processes)."""
    g = guidance()
    assigned: list[ast.expr | None] = []
    for node in ast.parse(inspect.getsource(pb)).body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "LIVE_TOOL_GUIDANCE" for t in targets):
            assigned.append(node.value)
    assert len(assigned) == 1, "T-0131 C2: LIVE_TOOL_GUIDANCE is assigned once at module level"
    assert isinstance(assigned[0], ast.Constant) and assigned[0].value == g, (
        "T-0131 C2: LIVE_TOOL_GUIDANCE is a string literal (implicit concatenation of literals is one constant)"
    )
    tools = T.tools_field()
    a = builder(tools=tools, tool_guidance=g, thinking=True)
    b = builder(tools=tools, tool_guidance=g, thinking=False)
    assert system_text(a) == system_text(b), (
        "T-0131 C2: the guided system text does not change with thinking"
    )
    history = [
        HistoryTurn(
            user_text="what is this", assistant_spoken="A shop.", frame_phash=_frame("h").phash
        ),
    ]
    t1 = a.prefix([], frames=(_frame("a"),))
    t2 = a.prefix(history, frames=(_frame("b"),))
    t3 = b.prefix(history)
    assert t1.messages[0] == t2.messages[0] == t3.messages[0]
    assert t1.prefix_len >= 1 and t2.prefix_len >= 1, "the system message is in the cached prefix"


def test_native_instructions_unchanged() -> None:
    """C3: ``NATIVE_TOOL_INSTRUCTIONS`` is BASE's (its pinned sha256), and in the guided system message it stands
    whole, directly before the separator and the guidance (nothing is spliced into it)."""
    g = guidance()
    native = pb.NATIVE_TOOL_INSTRUCTIONS
    assert hashlib.sha256(native.encode()).hexdigest() == T.NATIVE_TOOL_INSTRUCTIONS_SHA256
    guided = system_text(builder(tools=T.tools_field(), tool_guidance=g))
    head = guided[: len(guided) - len(g)].rstrip()
    assert head.endswith(native), (
        "T-0131 C3: the native instructions end the text before the guidance, unchanged"
    )
    assert native not in g and g not in native


def test_length_bound() -> None:
    """C1: at most 400 code points (and not empty: two sentences)."""
    g = guidance()
    assert 0 < len(g) <= 400, f"T-0131 C1: LIVE_TOOL_GUIDANCE is <= 400 code points; it is {len(g)}"
    assert g == g.strip(), "no leading or trailing whitespace (the separator is the builder's)"


def test_mentions_check_and_scroll() -> None:
    """C1 (keywords; the reviewer judges the wording): (a) check the post-action screenshot before saying an action
    worked, (b) scroll to see the rest of the page before answering about a whole page."""
    low = guidance().lower()
    assert "screenshot" in low, low
    assert any(w in low for w in ("check", "verify", "look at", "make sure")), low
    assert any(w in low for w in ("worked", "success", "succeed", "done")), low
    assert "scroll" in low and "page" in low, low


def test_no_confirmation_or_narration_request() -> None:
    """C1 (adversarial; the speaking-policy and autonomy rulings: ask only for pay, send or delete; no per-step
    narration): the guidance asks for no confirmation, no permission and no narration of each step."""
    low = guidance().lower()
    banned = (
        # word stems (ruling at 21361bd): "confirm" covers confirmation/confirm with/confirmed, "approv" approve/
        # approval/approved
        "confirm",
        "approv",
        "wait for",
        "each action",
        "every action",
        "before each",
        "before every",
        "tell the user",
        "aloud",
        "out loud",
        "permission",
        "ask the user",
        "ask for",
        "ask before",
        "ask first",
        "check with the user",
        "narrat",
        "announce",
        "each step",
        "every step",
        "describe each",
        "describe every",
        "tell the user what",
        "say what you",
    )
    found = [b for b in banned if b in low]
    assert not found, (
        f"T-0131 C1: the guidance must not ask for confirmation or narration; found {found}"
    )
