"""T-0201 group B, the pure half: ``toolcall.loop.with_screen_reminder`` and ``SCREEN_REMINDER_SEPARATOR``, and the
reminder table ``conductor.session.SCREEN_REMINDERS`` (docs/tasks/T-0201.md B2, B3, B10; H-T201-1, H-T201-2).

**Names pinned here** (the task leaves the helper's name open; these are the names group C's harness at 521468e4
already looks up, ``B_HELPER``/``B_TABLE``/``B_SEPARATOR``):

- ``loop.SCREEN_REMINDER_SEPARATOR == "\\n\\n"`` (H-T201-2);
- ``loop.with_screen_reminder(prompt: Prompt, reminder: str) -> Prompt``: append-only, ``SEPARATOR + reminder``
  after the continued final assistant message's text; ``ValueError`` on a prompt that does not continue a final
  assistant message (the ``with_report_fact`` contract, T-0178 B1);
- ``session.SCREEN_REMINDERS: Mapping[str, str]`` with exactly the five keys of B3, values H-T201-1's wording
  (typed below by hand from the task file, and pinned again by sha256 so a typo in this module cannot hide).

**Readings** (in the test-author's report):

- R-L1 (adversarial row, "continuation ending inside a payload string or containing ``<|"|>``"): the helper is
  append-only and does not interpret the continuation. A continuation holding ``<|"|>`` delimiters, a whole
  rendered response, or one that ends inside an open payload string comes back with every byte kept and exactly
  ``SEPARATOR + reminder`` after it (no sanitising of the existing text, no repair, no closing). Placement "after the
  last tool response" is the session's job (it calls the helper only on a step request), pinned in
  ``test_t0201_session_reminder.py``.
- R-L2 (B10 "``loop.__all__`` additions only"): BASE's ``__all__`` (read from the BASE blob, origin-proved) is a
  subset of today's, and today's adds exactly the two names above. Every other top-level definition of
  ``loop.py`` at BASE has the same AST at TIP (B10: payload builders, ``render_response``, ``REPORT_FACT``). Both
  rows compare the literal range BASE 620b1245 .. TIP 6d8e0a1d (group B's approved implementation commit), never
  the working tree (tests.md; closed in review 2026-10-29).

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor import session as session_mod
from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.toolcall import loop
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS, drop_special_literals
from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
#: BASE: main at 620b1245 (``T-0201 A merged``), where group B branched. A literal, never a moving ref (tests.md).
BASE_SHA = "620b1245"
#: TIP: group B's implementation commit 6d8e0a1d (approved in review); the B10 guards compare BASE..TIP (tests.md)
TIP_SHA = "6d8e0a1d"

#: H-T201-2, typed from the task file
SEP = "\n\n"
#: H-T201-1, typed from the task file (the key is B3's)
H_T201_1: dict[str, str] = {
    "now": "[You have the current screenshot: the last image in the user's message above is the browser screen now.]",
    "updated": (
        "[You have the current screenshot: the last image in the user's message above is the browser screen now, "
        "after your last action.]"
    ),
    "unchanged": (
        "[You have the current screenshot: the last image in the user's message above is the browser screen now; "
        "your last action did not change it.]"
    ),
    "shared_screen": (
        "[You have the current image: the last image in the user's message above is the user's shared screen now, "
        "not your browser.]"
    ),
    "shared_camera": (
        "[You have the current image: the last image in the user's message above is the user's camera now, not "
        "your browser.]"
    ),
}
#: sha256 of ``json.dumps(H_T201_1, sort_keys=True, ensure_ascii=True, separators=(",", ":"))``, computed once by
#: hand from the typed table above (B3: "table pinned by sha256")
TABLE_SHA256 = "433658d169afff665a8e65fb78108f54925b4995ea35151a4c051eaf4c3b855c"
#: sha256 of ``"\n\n"``
SEP_SHA256 = "75a11da44c802486bc6f65640aa48a730f0f684c5c07a42ba3cd1735eb3fb070"
#: what the task names the stem of every reminder (a substring of no other rig text)
STEM = "[You have the current"
#: B7: the room compaction reserves on a non-report step
MAX_ROOM = max(len(SEP + r) for r in H_T201_1.values())


def table_sha(table: Mapping[str, str]) -> str:
    blob = json.dumps(dict(table), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------------------------- the names


def helper() -> Any:
    fn = getattr(loop, "with_screen_reminder", None)
    assert callable(fn), (
        "T-0201 B2: local_astra/toolcall/loop.py defines with_screen_reminder(prompt, reminder) (the append helper)"
    )
    return fn


def separator() -> str:
    sep = getattr(loop, "SCREEN_REMINDER_SEPARATOR", None)
    assert isinstance(sep, str), (
        "T-0201 B2: loop.SCREEN_REMINDER_SEPARATOR (a str) does not exist yet"
    )
    return sep


def table() -> Mapping[str, str]:
    got = getattr(session_mod, "SCREEN_REMINDERS", None)
    assert isinstance(got, Mapping), (
        "T-0201 B3: conductor/session.py defines SCREEN_REMINDERS (a mapping key -> reminder text) beside LABEL_*"
    )
    return got


# --------------------------------------------------------------------------------------------- prompts


IMG = ImagePart(url="data:image/jpeg;base64,/9j/AAAA", phash="capture:00ff")
STEP1 = Prompt(
    messages=(
        Message(role="system", content=(TextPart(text="You are a test assistant."),)),
        Message(
            role="user",
            content=(
                IMG,
                TextPart(text="[This screenshot is the browser screen now.]"),
                TextPart(text="open the menu"),
            ),
        ),
    ),
    prefix_len=1,
    thinking=True,
)
STEPS = (
    loop.LoopStep(
        text="I will open the menu.",
        name="click_at",
        arguments={"x": 500, "y": 250},
        response={"action": "clicked at (640, 200)", "screenshot": "updated", "status": "executed"},
    ),
    loop.LoopStep(
        text="",
        name="scroll",
        arguments={"direction": "down"},
        response={
            "action": "scrolled down by 600 px",
            "screenshot": "updated",
            "status": "executed",
        },
    ),
)


def step_request() -> Prompt:
    return loop.step_prompt(STEP1, STEPS)


def with_tail(prompt: Prompt, text: str) -> Prompt:
    """``prompt`` (a continued step request) with its final assistant text replaced by ``text``."""
    return prompt.model_copy(
        update={
            "messages": (
                *prompt.messages[:-1],
                Message(role="assistant", content=(TextPart(text=text),)),
            )
        }
    )


def assert_appended(got: Prompt, base: Prompt, reminder: str) -> None:
    """B2: ``got`` is ``base`` with ``"\\n\\n" + reminder`` appended to its final assistant text; every other field,
    message and part is ``base``'s."""
    assert isinstance(got, Prompt), type(got)
    for field in Prompt.model_fields:
        if field != "messages":
            assert getattr(got, field) == getattr(base, field), (
                f"T-0201 B2: Prompt.{field} changed: {getattr(base, field)!r} -> {getattr(got, field)!r}"
            )
    assert len(got.messages) == len(base.messages), "T-0201 B2: no message added or removed"
    assert got.messages[:-1] == base.messages[:-1], "T-0201 B2: every earlier message is unchanged"
    last, was = got.messages[-1], base.messages[-1]
    assert last.role == was.role == "assistant"
    assert len(last.content) == 1 and isinstance(last.content[0], TextPart), last.content
    assert last.text() == was.text() + SEP + reminder, (
        f"T-0201 B2: append-only, SEPARATOR + reminder after the continuation\n"
        f"  want ...{(was.text() + SEP + reminder)[-200:]!r}\n  got  ...{last.text()[-200:]!r}"
    )


# =====================================================================================================
# the separator, the table
# =====================================================================================================


def test_separator_is_h_t201_2() -> None:
    """B2 / H-T201-2: the separator is two newlines (and the typed constant is that, by sha)."""
    assert hashlib.sha256(SEP.encode()).hexdigest() == SEP_SHA256
    assert separator() == SEP, f"T-0201 H-T201-2: SCREEN_REMINDER_SEPARATOR is {separator()!r}"


def test_table_is_h_t201_1_pinned_by_sha() -> None:
    """B3 / H-T201-1: exactly the five keys, each value the ruled wording; the whole table hashes to the pin."""
    assert table_sha(H_T201_1) == TABLE_SHA256, (
        "the typed table in this module drifted from its pin"
    )
    got = dict(table())
    assert set(got) == set(H_T201_1), (
        f"T-0201 B3: the keys are {sorted(H_T201_1)}; got {sorted(got)}"
    )
    for key, want in H_T201_1.items():
        assert got[key] == want, f"T-0201 H-T201-1 ({key}): {got[key]!r}"
    assert table_sha(got) == TABLE_SHA256


def test_reminders_are_plain_bracketed_facts() -> None:
    """B3: ASCII, bracketed, no special-token literal (``drop_special_literals(r) == r``), no page text, URL,
    coordinates, placeholder or clause (the outcome clause and the extent clause stems are absent), distinct, and
    each one opens with the reminder stem."""
    got = dict(table())
    assert len(set(got.values())) == len(got), "T-0201 B3: five distinct reminders"
    for key, r in got.items():
        assert r.isascii(), f"T-0201 B3 ({key}): not ASCII"
        assert r.startswith("[") and r.endswith("]") and r.count("[") == r.count("]") == 1, r
        assert r.startswith(STEM), r
        assert drop_special_literals(r) == r, f"T-0201 B3 ({key}): a special-token literal"
        assert [lit for lit in SPECIAL_TOKENS if lit in r] == [], r
        assert not any(ch.isdigit() for ch in r), (
            f"T-0201 B3 ({key}): no number or coordinate; {r!r}"
        )
        for banned in ("http", "www.", "://", "{", "}", "%s", "Compared with", "Page beyond", "\n"):
            assert banned not in r, f"T-0201 B3 ({key}): {banned!r} in {r!r}"
        for marker in (
            "<|tool_response>",
            "<tool_response|>",
            "<|tool_call>",
            "call:",
            "response:",
        ):
            assert marker not in r, f"T-0201 B3 ({key}): outside any payload; {marker!r}"


# =====================================================================================================
# the helper
# =====================================================================================================


@pytest.mark.parametrize("key", sorted(H_T201_1))
def test_helper_is_append_only(key: str) -> None:
    """B2: a two-step request through the helper is the request with ``"\\n\\n" + reminder`` after its last tool
    response; the system message, the user message (image, label, question), ``prefix_len``, ``thinking``, ``tools``,
    ``continue_final_message`` and every other field are unchanged. The input is not mutated."""
    base = step_request()
    snapshot = base.model_dump()
    assert base.messages[-1].text().endswith("<tool_response|>"), "premise: a step request"
    got = helper()(base, H_T201_1[key])
    assert_appended(got, base, H_T201_1[key])
    assert got.continue_final_message is True
    assert base.model_dump() == snapshot, "T-0201 B2: pure, the input prompt is not mutated"
    assert got.messages[-1].text().startswith(base.messages[-1].text()), (
        "the continuation is a prefix"
    )


def test_helper_twice_is_two_appends() -> None:
    """Append-only, composed: no de-duplication or replacement of an earlier suffix (the session never does this; the
    row pins that the helper is a plain append)."""
    base = step_request()
    once = helper()(base, H_T201_1["now"])
    twice = helper()(once, H_T201_1["updated"])
    assert twice.messages[-1].text() == (
        base.messages[-1].text() + SEP + H_T201_1["now"] + SEP + H_T201_1["updated"]
    )


@pytest.mark.parametrize(
    "case",
    ["step1", "final_user", "no_messages", "not_continued_assistant"],
)
def test_helper_refuses_a_non_continued_prompt(case: str) -> None:
    """B2 (the ``with_report_fact`` contract): a prompt that does not continue a final assistant message is refused
    with ``ValueError``: the step-1 prompt (final user message, ``continue_final_message`` false), a continued prompt
    whose final message is a user message, an empty prompt, and an assistant-final prompt that is not continued."""
    base = step_request()
    prompts = {
        "step1": STEP1,
        "final_user": base.model_copy(update={"messages": base.messages[:-1]}),
        "no_messages": base.model_copy(update={"messages": ()}),
        "not_continued_assistant": base.model_copy(update={"continue_final_message": False}),
    }
    with pytest.raises(ValueError):
        helper()(prompts[case], H_T201_1["now"])


@pytest.mark.parametrize(
    "tail",
    [
        # a continuation whose responses carry <|"|> delimiters (every real one does) and a special literal in a
        # payload string the renderer kept
        'I will look.<|tool_call>call:google_search{query:<|"|>a <|"|> b<|"|>}<tool_call|>'
        '<|tool_response>response:google_search{results:[{snippet:<|"|>x<|"|>}]}<tool_response|>',
        # one that ends inside an open payload string (no real step does; the helper must not repair or close it)
        '<|tool_call>call:type_at{text:<|"|>half a string',
        # one that ends inside a response frame
        '<|tool_response>response:scroll{action:<|"|>scrolled',
        # thought-looking text left in a step text
        "<|channel>thought\nI am not sure<channel|>",
    ],
    ids=["delimiters", "open_string", "open_response", "thought_markers"],
)
def test_helper_keeps_every_byte_of_an_awkward_continuation(tail: str) -> None:
    """Adversarial (R-L1): the helper never sanitises, re-renders or closes what it is given. A continuation with
    ``<|"|>`` delimiters, ending inside a payload string, inside a response, or holding thought markers comes back as
    ``tail + "\\n\\n" + reminder`` exactly (a helper that ran ``drop_special_literals`` over the whole text, or
    re-rendered the steps, fails here)."""
    base = with_tail(step_request(), tail)
    got = helper()(base, H_T201_1["unchanged"])
    assert_appended(got, base, H_T201_1["unchanged"])


def test_helper_keeps_an_image_in_the_continued_turn_untouched() -> None:
    """B2 "every other byte, message, part": a prompt whose user message carries two images and two texts keeps them
    in order (the helper never touches the user message, the label or the current frame)."""
    second = ImagePart(url="data:image/jpeg;base64,/9j/BBBB", phash="capture:11ee")
    user = Message(
        role="user",
        content=(
            second,
            TextPart(text="overview"),
            IMG,
            TextPart(text="[label]"),
            TextPart(text="q"),
        ),
    )
    base = loop.step_prompt(STEP1.model_copy(update={"messages": (STEP1.messages[0], user)}), STEPS)
    got = helper()(base, H_T201_1["shared_camera"])
    assert_appended(got, base, H_T201_1["shared_camera"])
    assert got.messages[1].content == user.content


# =====================================================================================================
# B10: loop.__all__ additions only; BASE's definitions unchanged
# =====================================================================================================


def _loop_source(sha: str) -> str:
    """``loop.py`` at the literal commit ``sha`` (skips, never passes, when it is absent)."""
    gr.require_commits(sha)
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{sha}:local_astra/toolcall/loop.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _base_loop_source() -> str:
    return _loop_source(BASE_SHA)


def _tip_loop_source() -> str:
    """R-L2, closed (review 2026-10-29): group B's TIP, never the working tree, so the guard does not decay when a
    later task legitimately edits ``loop.py``."""
    gr.require_commits(BASE_SHA, TIP_SHA)
    assert gr.is_ancestor(BASE_SHA, TIP_SHA), f"{BASE_SHA} is not an ancestor of {TIP_SHA}"
    return _loop_source(TIP_SHA)


def _all_of(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            return list(value)
    raise AssertionError("no __all__")


def _top_level(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id != "__all__":
                    out[t.id] = ast.dump(node)
    return out


def test_loop_all_gains_exactly_the_two_names() -> None:
    """B10 / R-L2: BASE's ``__all__`` (from the BASE blob) is kept, and the additions are exactly
    ``SCREEN_REMINDER_SEPARATOR`` and ``with_screen_reminder``; both are importable names of the module."""
    base = _all_of(ast.parse(_base_loop_source()))
    assert "with_report_fact" in base and "with_screen_reminder" not in base, "premise: BASE's list"
    now = _all_of(ast.parse(_tip_loop_source()))
    assert set(base) <= set(now), (
        f"T-0201 B10: names removed from __all__: {sorted(set(base) - set(now))}"
    )
    assert set(now) - set(base) == {"SCREEN_REMINDER_SEPARATOR", "with_screen_reminder"}, (
        f"T-0201 B10: __all__ gains exactly the separator and the helper; added {sorted(set(now) - set(base))}"
    )
    for name in ("SCREEN_REMINDER_SEPARATOR", "with_screen_reminder"):
        assert hasattr(loop, name), name


def test_loop_base_definitions_unchanged() -> None:
    """B10 (negative) / R-L2: every top-level function, class and constant of ``loop.py`` at BASE (payload builders,
    ``render_response``, ``step_prompt``, ``with_report_fact``, ``REPORT_FACT``, compaction ...) has the same AST
    at TIP 6d8e0a1d (docstrings included); group B only adds. Power: the same comparison sees the module docstring's change
    (B's **Screen reminder** paragraph), which is why the docstring is left out of the definitions compared."""
    base_tree = ast.parse(_base_loop_source())
    now_tree = ast.parse(_tip_loop_source())
    base, now = _top_level(base_tree), _top_level(now_tree)
    assert {"render_response", "search_payload", "REPORT_FACT", "step_prompt", "compact"} <= set(
        base
    )
    changed = sorted(n for n in base if now.get(n) != base[n])
    assert changed == [], f"T-0201 B10: BASE definitions of loop.py changed or removed: {changed}"
    assert ast.get_docstring(now_tree) != ast.get_docstring(base_tree), (
        "T-0201 B (owned files): loop.py's docstring gains a **Screen reminder** paragraph"
    )
    assert "Screen reminder" in (ast.get_docstring(now_tree) or "")
