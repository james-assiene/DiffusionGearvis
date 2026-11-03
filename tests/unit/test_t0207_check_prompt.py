"""T-0207 A1.1, A1.2, A1.4 and A1.6: ``local_astra.toolcall.completion_check`` (docs/tasks/T-0207.md).

**Shape** (A1.2): ``check_prompt(sent, said, frame, *, report)`` equals ``announced_call.reask_prompt(sent, said)``
except that its final user message is ``(ImagePart(frame), TextPart(<instruction>))``, in that order;
``continue_final_message`` is False; every other ``Prompt`` field is ``sent``'s (the thinking flag included). The
instruction is ``CHECK_INSTRUCTION`` when ``report`` is False and ``CHECK_INSTRUCTION_REPORT`` when it is True. Every
expected message list here is also written out by hand from A1.2 and T-0172 A2 (``reask_prompt``'s documented shape),
so the rows do not only compare the module against the function it composes.

**Readings this module pins** (named in the test-author's report):

- ``ImagePart(frame)`` is the part the prompt builder and ``loop.swap_newest_image`` make for a frame:
  ``url = "data:image/jpeg;base64," + base64(frame.jpeg)``, ``phash = frame.phash`` (A1.2 does not spell the encoding
  out; the frame's identity is its phash, which A2.9's grounding reads).
- "built by composing ``reask_prompt``, never by copying it" is checked behaviourally: ``reask_prompt`` replaced by a
  spy (on ``announced_call`` and, if the module bound the name, on ``completion_check``) is called once with
  ``(sent, said)``, and a spy that marks its result's system message shows the mark in ``check_prompt``'s result.
- The instruction texts are H-T207-3's, typed here by hand and pinned by sha256 (A1.4).

Adversarial (the task's failure mode, a prefix that is not one): continuations ending inside an open ``<|"|>``
payload, holding ``REPORT_FACT``, holding a T-0201 reminder, or carrying special literals keep every byte of the
continuation as a prefix of the check's final assistant message (and a Hypothesis property over arbitrary text).

A1.6 (negative): the guarded paths are byte-identical over the closed range BASE c97e5b30 .. TIP 5e947d3e (the A1
implementation tip, pinned after review), on any branch, never the working tree; ancestry is proved, and the row
skips naming the SHA if either commit is absent. A power row proves the same predicate sees a change it must see.

Regime: CPU, pure; deterministic (Hypothesis seeded); n = 1.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.contracts.perception import AssistFlags, Frame, Source
from local_astra.toolcall import announced_call, loop
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS, drop_special_literals
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.toolcall.completion_check"

#: H-T207-3, typed from docs/tasks/T-0207.md
H_CHECK = (
    "This is the browser screen now. Before anything reaches the user, check what you just wrote against this "
    "screen. If the task is not done, continue with a tool call. If it is done, answer the user again in one or two "
    "sentences, saying only what this screen shows."
)
H_CHECK_REPORT = (
    "This is the browser screen now. You have no actions left. Check what you just wrote against this screen, then "
    "tell the user in one or two sentences what this screen shows is done and where you stopped. Do not call a tool."
)
#: sha256 of the two texts above (computed by hand from the task file's quoted text, 2026-10-30)
SHA_CHECK = "2d03797ff58ee9e65604a16be9b9857731a91153fbbedaa825b8aea8b8e29acc"
SHA_CHECK_REPORT = "5e28b1784e95e17fde8c0a6068281bb799503fd22ba8a314eb130c9258e8638a"
#: T-0201 H-T201-1's "unchanged" reminder, typed (conductor.session.SCREEN_REMINDERS is not imported here)
REMINDER = (
    "[You have the current screenshot: the last image in the user's message above is the browser screen "
    "now; your last action did not change it.]"
)

SAID = "I have added volume 7 of Vinland Saga to your cart."
SYSTEM = "You are a synthetic test assistant that can use a browser."
STEP_JPEG = b"\xff\xd8\xff\xe0step-frame"
FRESH = Frame(
    source=Source.WINDOW,
    ts_ms=7,
    width=1280,
    height=800,
    jpeg=b"\xff\xd8\xff\xe0fresh-frame-bytes",
    phash="0123456789abcdef",
)
TOOLS: tuple[dict[str, Any], ...] = tuple(
    {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
    for n in ("click_at", "scroll")
)

#: A1.6: the paths that stay byte-identical over BASE..TIP
BASE_SHA = "c97e5b30"  # main when T-0207 was planned; the T-0207-a1 branch's fork point
TIP_SHA: str | None = "5e947d3e"  # T-0207 A1 implementation tip (pinned 2026-10-30, after review)
GUARDED = (
    "local_astra/toolcall/announced_call.py",
    "local_astra/toolcall/loop.py",
    "local_astra/conductor/session.py",
    "local_astra/conductor/turn_machine.py",
    "local_astra/contracts",
    "local_astra/toolcall/tools_v1.json",
    "local_astra/toolcall/tools_v1_overview.json",
    "local_astra/toolcall/tools_v1_overview_wait.json",
    "local_astra/toolcall/tools_v1_wait.json",
    "scripts/replay_announced_calls.py",
    "scripts/replay_degeneracy.py",
    "scripts/replay_implicit_thought.py",
    "scripts/replay_screen_reminder.py",
    "scripts/replay_thought_calls.py",
    "scripts/replay_ui_gate.py",
)


def _mod() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE:
            raise NotImplementedError(f"T-0207 A1.1: {MODULE} does not exist yet") from exc
        raise


def fresh_part(frame: Frame = FRESH) -> ImagePart:
    return ImagePart(
        url="data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode(), phash=frame.phash
    )


def step1(*, thinking: bool = True, tools: tuple[dict[str, Any], ...] | None = TOOLS) -> Prompt:
    return Prompt(
        messages=(
            Message(role="system", content=(TextPart(text=SYSTEM),)),
            Message(
                role="user",
                content=(
                    ImagePart(
                        url="data:image/jpeg;base64," + base64.b64encode(STEP_JPEG).decode(),
                        phash="capture:aaaa",
                    ),
                    TextPart(
                        text="[This screenshot is the browser screen now, after your last action.]"
                    ),
                    TextPart(text="buy volume 7 of the synthetic comic"),
                ),
            ),
        ),
        prefix_len=2,
        thinking=thinking,
        assists=AssistFlags(),
        tools=tools,
    )


def steps(n: int) -> list[loop.LoopStep]:
    out = [
        loop.LoopStep(
            text="I will open the shop.",
            name="click_at",
            arguments={"x": 485, "y": 281},
            response={
                "action": "clicked at (485, 281)",
                "page": "no new page loaded",
                "status": "executed",
            },
        )
    ]
    out += [
        loop.LoopStep(
            text="",
            name="scroll",
            arguments={"direction": "down"},
            response={
                "action": "scrolled down by 600 px",
                "screenshot": "updated",
                "status": "executed",
            },
        )
        for _ in range(n - 1)
    ]
    return out


def step_k(k: int, base: Prompt | None = None) -> Prompt:
    """The step-``k`` request (k >= 2): ``loop.step_prompt`` over ``k - 1`` executed steps."""
    return loop.step_prompt(base or step1(), steps(k - 1))


def report_step(k: int = 5) -> Prompt:
    return loop.with_report_fact(step_k(k))


def reminder_step(k: int = 3) -> Prompt:
    return loop.with_screen_reminder(step_k(k), REMINDER)


def other_fields(p: Prompt) -> dict[str, Any]:
    return {
        f: getattr(p, f)
        for f in Prompt.model_fields
        if f not in ("messages", "continue_final_message")
    }


def instruction(m: ModuleType, report: bool) -> str:
    return str(m.CHECK_INSTRUCTION_REPORT if report else m.CHECK_INSTRUCTION)


def by_hand(sent: Prompt, said: str, text: str, frame: Frame = FRESH) -> tuple[Message, ...]:
    """A1.2 written out from T-0172 A2's documented shape, without calling ``reask_prompt``."""
    if sent.continue_final_message:
        head = sent.messages[:-1]
        said = sent.messages[-1].text() + said
    else:
        head = sent.messages
    return (
        *head,
        Message(role="assistant", content=(TextPart(text=said),)),
        Message(role="user", content=(fresh_part(frame), TextPart(text=text))),
    )


SENTS = {
    "step1": step1,
    "step2": lambda: step_k(2),
    "step5": lambda: step_k(5),
    "report": report_step,
    "reminder": reminder_step,
}


# --------------------------------------------------------------------------------------------- A1.1 surface


def test_module_surface() -> None:
    """A1.1: two string constants and ``check_prompt(sent, said, frame, *, report) -> Prompt``."""
    m = _mod()
    assert isinstance(m.CHECK_INSTRUCTION, str) and isinstance(m.CHECK_INSTRUCTION_REPORT, str)
    sig = inspect.signature(m.check_prompt)
    params = list(sig.parameters.values())
    assert [p.name for p in params[:3]] == ["sent", "said", "frame"], sig
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params[:3]), sig
    assert (
        "report" in sig.parameters
        and sig.parameters["report"].kind is inspect.Parameter.KEYWORD_ONLY
    ), sig
    # T-0227 R-3b re-pin (named, docs/tasks/T-0227.md R-3): a fifth parameter, keyword-only ``max_images``
    assert (
        "max_images" in sig.parameters
        and sig.parameters["max_images"].kind is inspect.Parameter.KEYWORD_ONLY
    ), sig
    assert [p.name for p in params] == ["sent", "said", "frame", "report", "max_images"], sig
    assert len(params) == 5, f"A1.1 + T-0227 R-3 name exactly five parameters: {sig}"
    with pytest.raises(TypeError):
        m.check_prompt(step_k(2), SAID, FRESH, False)  # report is keyword-only
    with pytest.raises(TypeError):
        m.check_prompt(step_k(2), SAID, FRESH, False, 4)  # so is max_images


def test_t0227_default_max_images_is_the_builders() -> None:
    """T-0227 R-3b parity: ``completion_check.DEFAULT_MAX_IMAGES`` (a literal: the module may not import the prompt
    builder) equals ``prompt_builder.Budget().max_images``, and it is ``check_prompt``'s default for ``max_images``."""
    from local_astra.conductor.prompt_builder import Budget

    m = _mod()
    assert type(m.DEFAULT_MAX_IMAGES) is int and Budget().max_images == m.DEFAULT_MAX_IMAGES
    assert (
        inspect.signature(m.check_prompt).parameters["max_images"].default == m.DEFAULT_MAX_IMAGES
    )


def test_imports_stdlib_contracts_and_announced_call_only() -> None:
    """A1.1: "stdlib, ``local_astra.contracts`` and ``toolcall.announced_call`` only"."""
    m = _mod()
    tree = ast.parse(Path(str(m.__file__)).read_text())
    bad = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import in {MODULE}"
            mod = node.module or ""
            # ``from local_astra.toolcall import announced_call`` imports exactly that module
            names = (
                [f"{mod}.{a.name}" for a in node.names] if mod == "local_astra.toolcall" else [mod]
            )
        for n in names:
            top = n.split(".")[0]
            if n == "local_astra.contracts" or n.startswith("local_astra.contracts."):
                continue
            if n == "local_astra.toolcall.announced_call":
                continue
            if top in sys.stdlib_module_names or top == "__future__":
                continue
            bad.append(n)
    assert not bad, f"{MODULE} imports outside stdlib, contracts and announced_call: {bad}"


# --------------------------------------------------------------------------------------------- A1.4


def test_instruction_texts_pinned() -> None:
    """A1.4 / H-T207-3: the two texts, exactly, and their sha256."""
    m = _mod()
    assert m.CHECK_INSTRUCTION == H_CHECK, m.CHECK_INSTRUCTION
    assert m.CHECK_INSTRUCTION_REPORT == H_CHECK_REPORT, m.CHECK_INSTRUCTION_REPORT
    assert hashlib.sha256(m.CHECK_INSTRUCTION.encode()).hexdigest() == SHA_CHECK
    assert hashlib.sha256(m.CHECK_INSTRUCTION_REPORT.encode()).hexdigest() == SHA_CHECK_REPORT
    assert m.CHECK_INSTRUCTION != m.CHECK_INSTRUCTION_REPORT


@pytest.mark.parametrize("report", [False, True], ids=["check", "report"])
def test_instruction_ascii_no_special_no_url_no_coordinates(report: bool) -> None:
    """A1.4: ASCII; ``drop_special_literals(x) == x``; no URL, page text or coordinates (no digit at all)."""
    m = _mod()
    x = instruction(m, report)
    assert x.isascii(), [c for c in x if not c.isascii()]
    assert drop_special_literals(x) == x
    assert not any(lit in x for lit in SPECIAL_TOKENS)
    assert "<" not in x and ">" not in x and "|" not in x, "no token-like markup"
    assert not any(c.isdigit() for c in x), "no coordinate or number"
    low = x.lower()
    for url_bit in ("http", "://", "www.", ".com", ".org"):
        assert url_bit not in low, url_bit
    assert x.strip() == x and x, repr(x)


# --------------------------------------------------------------------------------------------- A1.2 shape


@pytest.mark.parametrize("report", [False, True], ids=["check", "report"])
@pytest.mark.parametrize("name", list(SENTS))
def test_shape_equals_reask_with_last_user_replaced(name: str, report: bool) -> None:
    """A1.2: ``reask_prompt(sent, said)`` with its final user message replaced by ``(ImagePart(frame),
    TextPart(<instruction>))``; the same message list written out by hand."""
    m = _mod()
    sent = SENTS[name]()
    got = m.check_prompt(sent, SAID, FRESH, report=report)
    assert isinstance(got, Prompt)
    reask = announced_call.reask_prompt(sent, SAID)
    want_last = Message(role="user", content=(fresh_part(), TextPart(text=instruction(m, report))))
    assert got.messages[:-1] == reask.messages[:-1], got.messages[-3:]
    assert got.messages[-1] == want_last, got.messages[-1]
    assert got.messages == by_hand(sent, SAID, H_CHECK_REPORT if report else H_CHECK)
    assert got.continue_final_message is False
    assert other_fields(got) == other_fields(reask)


@pytest.mark.parametrize("report", [False, True], ids=["check", "report"])
def test_image_before_text_and_frame_identity(report: bool) -> None:
    """A1.2: the final user message is exactly two parts, the image first; the image is the given frame (its jpeg
    and phash), never the step's image."""
    m = _mod()
    sent = step_k(3)
    got = m.check_prompt(sent, SAID, FRESH, report=report)
    last = got.messages[-1]
    assert last.role == "user" and len(last.content) == 2, last
    img, text = last.content
    assert isinstance(img, ImagePart) and isinstance(text, TextPart), last.content
    assert img.phash == FRESH.phash
    assert img.url == "data:image/jpeg;base64," + base64.b64encode(FRESH.jpeg).decode()
    assert text.text == instruction(m, report)
    other = FRESH.model_copy(update={"jpeg": b"\xff\xd8other", "phash": "fedcba9876543210"})
    img2 = m.check_prompt(sent, SAID, other, report=report).messages[-1].content[0]
    assert img2 == fresh_part(other) and img2 != img


@pytest.mark.parametrize(
    "sent",
    [
        step1(thinking=True),
        step1(thinking=False),
        step1(tools=None),
        Prompt(
            messages=(Message(role="user", content=(TextPart(text="Find the form."),)),),
            prefix_len=0,
            thinking=True,
            assists=AssistFlags(page_text=True, pdf_text=True),
            tools=TOOLS[:1],
        ),
    ],
    ids=["thinking", "no-thinking", "no-tools", "assists-no-system"],
)
@pytest.mark.parametrize("continued", [False, True], ids=["step1", "step3"])
@pytest.mark.parametrize("report", [False, True], ids=["check", "report"])
def test_other_fields_equal(sent: Prompt, continued: bool, report: bool) -> None:
    """A1.2: every other ``Prompt`` field equals ``sent``'s, including the thinking flag; the leading messages are
    ``sent``'s, untouched."""
    m = _mod()
    if continued:
        sent = step_k(3, sent)
    got = m.check_prompt(sent, SAID, FRESH, report=report)
    assert other_fields(got) == other_fields(sent), (other_fields(got), other_fields(sent))
    assert got.thinking is sent.thinking
    assert got.continue_final_message is False
    assert got.messages[: len(sent.messages) - 1] == sent.messages[:-1]
    assert len(got.messages) == len(sent.messages) + (1 if continued else 2)


def test_composes_reask_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A1.2: built by composing ``reask_prompt``, never by copying it. A spy is called once with ``(sent, said)``,
    and what it returns (here: a marked system message) is what the check prompt is built on."""
    m = _mod()
    real = announced_call.reask_prompt
    calls: list[tuple[Any, ...]] = []

    def spy(sent: Prompt, said: str) -> Prompt:
        calls.append((sent, said))
        out = real(sent, said)
        marked = Message(role="system", content=(TextPart(text="MARKED BY THE SPY"),))
        return out.model_copy(update={"messages": (marked, *out.messages[1:])})

    monkeypatch.setattr(announced_call, "reask_prompt", spy)
    if hasattr(m, "reask_prompt"):
        monkeypatch.setattr(m, "reask_prompt", spy)
    sent = step_k(4)
    got = m.check_prompt(sent, SAID, FRESH, report=False)
    assert calls == [(sent, SAID)], calls
    assert got.messages[0].text() == "MARKED BY THE SPY", (
        "the result is not built on reask_prompt's"
    )


def test_sent_is_not_mutated() -> None:
    m = _mod()
    for make in SENTS.values():
        sent = make()
        before = sent.model_dump()
        m.check_prompt(sent, SAID, FRESH, report=True)
        m.check_prompt(sent, SAID, FRESH, report=False)
        assert sent.model_dump() == before


@pytest.mark.parametrize("report", [False, True], ids=["check", "report"])
def test_instruction_only_in_last_message(report: bool) -> None:
    """Adversarial: the instruction is in exactly one message, the last; the system message is ``sent``'s; neither
    instruction appears anywhere else, and the re-ask nudge is gone."""
    m = _mod()
    sent = report_step() if report else step_k(3)
    got = m.check_prompt(sent, SAID, FRESH, report=report)
    for text in (H_CHECK, H_CHECK_REPORT):
        holders = [i for i, msg in enumerate(got.messages) if text in msg.text()]
        assert holders == ([len(got.messages) - 1] if text == instruction(m, report) else []), (
            text,
            holders,
        )
    assert got.messages[0] == sent.messages[0] and got.messages[0].role == "system"
    assert announced_call.REASK_NUDGE not in "".join(msg.text() for msg in got.messages)


# --------------------------------------------------------------------------------------------- adversarial prefix


def continued(text: str) -> Prompt:
    """A step request whose continuation is exactly ``text``."""
    base = step1()
    return base.model_copy(
        update={
            "messages": (*base.messages, Message(role="assistant", content=(TextPart(text=text),))),
            "continue_final_message": True,
        }
    )


ODD = {
    # ends inside an open string payload of a call
    "open-payload": 'I will type it.<|tool_call>call:type_at{text:<|"|>half a qu',
    # ends with REPORT_FACT after the last response
    "report-fact": loop.with_report_fact(step_k(3)).messages[-1].text(),
    # ends with a reminder after the last response
    "reminder": loop.with_screen_reminder(step_k(3), REMINDER).messages[-1].text(),
    # both trailing whitespace and a closing delimiter
    "trailing-ws": 'response:click_at{status:<|"|>executed<|"|>}<tool_response|>\n\n  ',
}


@pytest.mark.parametrize("report", [False, True], ids=["check", "report"])
@pytest.mark.parametrize("name", list(ODD))
def test_odd_continuations_keep_their_prefix_bytes(name: str, report: bool) -> None:
    """Adversarial (A1.3's failure mode at the message level): the final assistant message of the check prompt starts
    with the continuation's exact bytes (nothing closed, trimmed, sanitised or reordered) and continues with
    ``said``; ``REPORT_FACT`` or the reminder stays where the step put it, once."""
    m = _mod()
    text = ODD[name]
    sent = continued(text)
    got = m.check_prompt(sent, SAID, FRESH, report=report)
    final = got.messages[-2]
    assert final.role == "assistant" and len(final.content) == 1
    assert final.text() == text + SAID, (final.text()[-120:], (text + SAID)[-120:])
    if name == "report-fact":
        assert final.text().count(loop.REPORT_FACT) == 1
    if name == "reminder":
        assert final.text().count(REMINDER) == 1


@seed(20761030)
@settings(max_examples=200, deadline=None, database=None)
@given(
    text=st.text(max_size=60),
    tail=st.lists(st.sampled_from(['<|"|>', "x", " ", "\n", "<tool_call|>", "Z"]), max_size=12).map(
        "".join
    ),
    said=st.text(max_size=40),
    report=st.booleans(),
)
def test_prefix_bytes_property(text: str, tail: str, said: str, report: bool) -> None:
    """Hypothesis: for any continuation and any ``said``, the check prompt keeps ``sent``'s messages but the last,
    puts ``continuation + said`` in its final assistant message, and ends with the fresh frame and the instruction."""
    m = _mod()
    cont = (text + tail) or "x"
    sent = continued(cont)
    got = m.check_prompt(sent, said, FRESH, report=report)
    assert got.messages[:-2] == sent.messages[:-1]
    assert got.messages[-2] == Message(role="assistant", content=(TextPart(text=cont + said),))
    assert got.messages[-1] == Message(
        role="user", content=(fresh_part(), TextPart(text=H_CHECK_REPORT if report else H_CHECK))
    )
    assert got.continue_final_message is False


# --------------------------------------------------------------------------------------------- A1.6 (negative)


def _changed_now(paths: tuple[str, ...]) -> list[str]:
    """BASE vs TIP when TIP is pinned, else BASE vs the working tree (tracked changes) on a T-0207-a1 branch only."""
    gr.require_commits(BASE_SHA)
    if TIP_SHA is not None:
        gr.require_commits(TIP_SHA)
        assert gr.is_ancestor(BASE_SHA, TIP_SHA), f"{BASE_SHA} is not an ancestor of {TIP_SHA}"
        return gr.changed(BASE_SHA, TIP_SHA, *paths)
    branch = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not branch.startswith("T-0207-a1"):
        pytest.skip(
            f"A1.6: TIP_SHA is not pinned yet and HEAD is on {branch!r}, not a T-0207-a1 branch"
        )
    proc = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--name-only", BASE_SHA, "--", *paths],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return sorted(proc.stdout.split())


def test_guarded_paths_byte_identical() -> None:
    """A1.6: ``announced_call.py``, ``loop.py``, ``session.py``, ``turn_machine.py``, ``contracts/**``,
    ``tools_v1*.json`` and every replay script that exists at BASE are unchanged."""
    gr.require_commits(BASE_SHA)
    listed = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "--name-only", BASE_SHA, "scripts/"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    at_base = sorted(p for p in listed if Path(p).name.startswith("replay_") and p.endswith(".py"))
    assert at_base == sorted(p for p in GUARDED if p.startswith("scripts/")), at_base
    assert _changed_now(GUARDED) == []


def test_guard_predicate_has_power() -> None:
    """Power row: the same predicate over a range known to change ``loop.py`` and ``session.py`` (T-0201 B,
    6d8e0a1d, against its parent) reports them."""
    gr.require_commits("6d8e0a1d")
    got = gr.changed("6d8e0a1d^", "6d8e0a1d", *GUARDED)
    assert "local_astra/toolcall/loop.py" in got and "local_astra/conductor/session.py" in got, got
