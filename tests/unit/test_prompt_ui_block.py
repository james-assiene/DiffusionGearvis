"""T-0044, the prompt-builder half (criteria 1, 6, 7, 11) + T-0055 criteria 7 and 11.

What is asserted here is *where* the block lives and what its presence must not disturb:

* criterion 1 — it is in the system message, exactly once, and in no other message of
  ``prefix()`` or ``full()``. The system turn is the cached prefix
  (``.claude/rules/conductor.md``: everything up to the frame is the prefix; memory, state and
  the transcript come after because they depend on the query), so a block that drifted into the
  suffix would be re-paid on every turn and would move the cache boundary;
* criterion 7 — the system bytes are identical between two builders and between two turns, and
  (the harder half, which no in-process comparison can see) between two *processes*: a block
  assembled from a ``frozenset`` would be re-ordered by string-hash randomisation and would miss
  the prefix cache on every session. That row runs two interpreters with different
  ``PYTHONHASHSEED`` and is deliberately **not** pending — it is green today and must stay green;
* criterion 6 — ``fit()``'s ladder (T-0020 ruling A1: history pixels → memory hits → whole turns)
  is untouched and the system message is the floor, not a rung;
* criterion 11 — ``SPOKEN_STYLE`` and ``CLOSE_THE_THOUGHT`` keep their wording and their place.

T-0055 adds two rows here and changes none of the above:

* criterion 7 — the system message is the four parts it is documented to be and **nothing else**,
  so "the cached prefix changes exactly once" is a statement about the block alone. The expected
  text is *derived* from ``PromptBuilder.system_message()``'s own inputs, never from a copy of the
  block's text (the amendment T-0055 relays to T-0053: a pin that copies the block asserts current
  behaviour captured from the implementation, which ``.claude/rules/tests.md`` forbids, and it
  would redden this module the day the block is rewritten);
* criterion 11 — **the human's ruling, as a source scan.** "This is only solvable by prompting a
  priori for us (we don't want to extract such things from that channel ourselves)." No harvesting
  a fence out of the reasoning stream, no client-side recovery, no promotion of a thought commit
  to a ``ui`` commit. The scan is what a future implementer meets when they reach for the obvious
  workaround.

Expected values: the trimming expectations are hand-computed from the caps in ``Budget``
(``max_images`` 4, ``image_age_turns`` 2, ``max_history_turns`` 3) and stated in the test that
uses them; the token counts come from ``FakeEngineTokenizer``, the structural stand-in for
``usage.prompt_tokens`` that ``test_prompt_budget.py`` already uses as its oracle (shared on
purpose: two stand-ins would be two truths). ``test_prompt_budget.py`` itself is untouched by
this task and the verifier re-runs it.
"""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from local_astra.conductor import prompt_builder as pb_module
from local_astra.conductor.prompt_builder import Budget, PromptBuilder
from local_astra.contracts.perception import AssistFlags
from tests.unit.test_prompt_budget import (
    TRANSCRIPT,
    FakeEngineTokenizer,
    frame,
    history,
    hits,
    resolver,
)

try:  # T-0044 creates it; a missing import must not error at collection
    from local_astra.conductor import ui_instructions as _ui_instructions
except ImportError:  # pragma: no cover - until T-0044 lands
    _ui_instructions = None

REPO = Path(__file__).resolve().parents[2]

SYSTEM = "You are Astra."

#: Criterion 11, verbatim. Pinned as literals — not read from the module — because the criterion
#: is precisely "these two keep their current wording": comparing the module against itself would
#: say nothing. ``.claude/rules/conductor.md``: the prompt always instructs closing the thought.
CLOSE_THE_THOUGHT_TODAY = (
    "Think step by step inside your thinking channel, then close it and give the spoken answer "
    "as plain sentences. Never leave the answer inside the thinking channel."
)
SPOKEN_STYLE_TODAY = (
    "You are speaking aloud: short sentences, no markdown, no lists, numbers as words when short. "
    "If you also show something, put it in the ui channel, never in speech."
)

#: T-0020 criterion 1 / ruling A5. Unchanged by this task: the block is spent out of the prompt
#: budget, never out of the answer's headroom.
MAX_PROMPT_TOKENS = 14_336
ANSWER_HEADROOM = 2_048


def ui_block() -> str:
    if _ui_instructions is None or getattr(_ui_instructions, "UI_INSTRUCTIONS", None) is None:
        raise NotImplementedError(
            "local_astra.conductor.ui_instructions.UI_INSTRUCTIONS does not exist: the cached "
            "system prefix still never mentions the A2UI channel (docs/tasks/T-0044.md)"
        )
    return str(_ui_instructions.UI_INSTRUCTIONS)


def builder(budget: Budget | None = None) -> PromptBuilder:
    return PromptBuilder(SYSTEM, budget)


def full_prompt(b: PromptBuilder, turns=None, memory=None):
    """A representative turn: 3 history turns with pixels, the current frame, 3 memory hits."""
    turns = history(3) if turns is None else turns
    memory = hits(0.9, 0.5, 0.2) if memory is None else memory
    prefix = b.prefix(turns, frame("c"), AssistFlags(), resolver(turns))
    return b.full(prefix, TRANSCRIPT, None, memory)


# --- criterion 11: what must not change --------------------------------------------------------


def test_the_spoken_style_and_close_the_thought_rules_survive():
    """Criterion 11 (current behaviour — not pending): both rules, verbatim, still in the prefix.

    The block is *added* to the system message; the two instructions that were already there keep
    their wording and their place. ``tests/unit/test_prompt_budget.py:212`` reads
    ``CLOSE_THE_THOUGHT`` out of the system message and would break silently if it moved.
    """
    assert pb_module.CLOSE_THE_THOUGHT == CLOSE_THE_THOUGHT_TODAY, (
        "criterion 11: CLOSE_THE_THOUGHT keeps its wording (X7 is the task that tunes it)"
    )
    assert pb_module.SPOKEN_STYLE == SPOKEN_STYLE_TODAY, (
        "criterion 11: SPOKEN_STYLE keeps its wording — including its last sentence, which is the "
        "only thing that pointed at the ui channel before this task"
    )
    text = builder().system_message().text()
    assert CLOSE_THE_THOUGHT_TODAY in text and SPOKEN_STYLE_TODAY in text
    assert (
        text.index(SYSTEM) < text.index(CLOSE_THE_THOUGHT_TODAY) < text.index(SPOKEN_STYLE_TODAY)
    ), "the order of the existing instructions is part of the cached prefix; only appending is free"


def test_the_budget_constants_are_unchanged():
    """Criterion 5c: the block is spent out of the prompt budget, not the answer's headroom.

    14 336 = 16 384 (``--max-model-len``) − 2 048 (ruling A5). An implementer who found the block
    made prompts too big could 'fix' it by moving one of these; that is a T-0020 change with its
    own task, not a side effect of a prompt edit.
    """
    assert Budget().max_prompt_tokens == MAX_PROMPT_TOKENS, Budget().max_prompt_tokens
    assert pb_module.ANSWER_HEADROOM_TOKENS == ANSWER_HEADROOM, pb_module.ANSWER_HEADROOM_TOKENS
    assert (Budget().max_images, Budget().image_age_turns, Budget().max_history_turns) == (4, 2, 3)


# --- criterion 7: the prefix is byte-identical --------------------------------------------------


def test_the_system_message_is_the_same_bytes_in_two_processes():
    """Criterion 7 across *sessions* (current behaviour — not pending, and must stay green).

    Two interpreters with different ``PYTHONHASHSEED`` build the system message and print its
    SHA-256. This is the row that catches the tempting implementation the task's notes warn
    against: a block assembled at import time from ``ADVERTISED_COMPONENTS`` (a ``frozenset``)
    iterates in an order that depends on the process's string-hash seed, so every session would
    get a different cached prefix and the *first* prompt of every session would miss the prefix
    cache for a reason nobody would ever find in-process.
    """
    code = (
        "import hashlib;"
        "from local_astra.conductor.prompt_builder import PromptBuilder;"
        f"print(hashlib.sha256(PromptBuilder({SYSTEM!r}).system_message()"
        ".text().encode()).hexdigest())"
    )
    digests = [
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": seed, "HOME": os.environ["HOME"]},
        ).stdout.strip()
        for seed in ("0", "1")
    ]
    assert digests[0] == digests[1], (
        "the system message differs between two processes with different PYTHONHASHSEED: the "
        "cached prefix is not stable across sessions. Build the prompt text from literals and "
        "ordered sequences, never by iterating a set (T-0044 criterion 7, implementation notes)"
    )


def test_two_builders_and_two_turns_produce_the_same_system_bytes():
    """Criterion 7 in-process: same config → same bytes, and the block is inside those bytes.

    The prefix cache is the reason ``<|think|>`` is a session setting and thinking is never
    toggled per request (CLAUDE.md "Observed mistakes"); a system message that varied per turn
    would cost a full prefill every turn.
    """
    first, second = builder(), builder()
    assert first.system_message() == second.system_message()

    turn_one = full_prompt(first)
    turn_two = full_prompt(first, turns=history(3)[:2])
    assert turn_one.messages[0].text().encode() == turn_two.messages[0].text().encode(), (
        "two consecutive turns of one builder produced different system bytes"
    )
    assert ui_block() in turn_one.messages[0].text(), (
        "criterion 1/7: the A2UI block is part of the bytes that must stay identical"
    )
    assert (
        hashlib.sha256(first.system_message().text().encode()).hexdigest()
        == hashlib.sha256(second.system_message().text().encode()).hexdigest()
    )


# --- criterion 1: exactly once, in the system message -------------------------------------------


def test_the_block_is_in_the_system_message_and_in_no_other_message():
    """Criterion 1: substring search over every message of a built prompt — count 1, index 0.

    Both halves matter. In the *system* message the block is prefilled once per session and
    cached; anywhere after the frame it is query-dependent text re-sent (and re-attended) every
    turn, which is the shape a well-meaning "put the instructions next to the transcript so the
    model remembers them" edit produces.
    """
    b = builder()
    prompt = full_prompt(b)
    block = ui_block()

    carrying = [i for i, m in enumerate(prompt.messages) if block in m.text()]
    assert carrying == [0], (
        f"the block must appear in message 0 and nowhere else; it appears in {carrying} of "
        f"{len(prompt.messages)} messages"
    )
    assert block in b.system_message().text(), "PromptBuilder.system_message() carries the block"

    # not even a fragment of it may leak into the query-dependent suffix
    lines = [line.strip() for line in block.splitlines() if len(line.strip()) >= 20]
    assert lines, "the block is one line of fewer than 20 characters? check the seam"
    for i, message in enumerate(prompt.messages[1:], start=1):
        leaked = [line for line in lines if line in message.text()]
        assert not leaked, (
            f"message {i} ({message.role}) repeats part of the block: {leaked[:1]} — the "
            "instructions belong to the cached prefix only (docs/02 §2.2)"
        )

    prefix_only = b.prefix(history(3), frame("c"), AssistFlags(), resolver(history(3)))
    assert [i for i, m in enumerate(prefix_only.messages) if block in m.text()] == [0]


# --- criterion 6: fit() never spends the block --------------------------------------------------


def test_fit_keeps_the_system_message_and_the_users_turn_at_a_budget_of_one():
    """T-0020's floor, pinned as it is today (not pending): the ladder has three rungs and the
    system message is not one of them.

    Hand-computed for the fixture below (3 history turns, each with a frame; the current frame;
    3 memory hits) against the caps in ``Budget``: ``image_age_turns`` 2 means only the two
    youngest history frames travel as pixels, so the images rung can spend exactly **2**; then
    the 3 memory hits; then the 3 history turns. Nothing else can be spent, so ``fit`` gives up
    with ``fits=False`` rather than truncating the user's question.
    """
    b = builder(Budget(max_prompt_tokens=1))
    turns = history(3)
    result = b.fit(
        history=turns,
        frame=frame("c"),
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=hits(0.9, 0.5, 0.2),
        resolve=resolver(turns),
        measure=FakeEngineTokenizer(),
    )

    assert result.fits is False, "1 token is not enough for the system message alone"
    assert dict(result.trimmed) == {"history_turns": 3, "images": 2, "memory_hits": 3}, dict(
        result.trimmed
    )
    assert pb_module.CLOSE_THE_THOUGHT in result.prompt.messages[0].text(), (
        "the system message is the floor, never a rung"
    )
    assert TRANSCRIPT in result.prompt.messages[-1].text(), (
        "criterion 4 of T-0020: the user's own turn is never truncated to make the prompt fit"
    )


def test_a_representative_prompt_still_fits_with_the_block():
    """Criterion 6, first half: the block does not push a normal turn over the budget.

    Regime of the number: ``FakeEngineTokenizer`` (4 tokens per message, 2 per whitespace word,
    1120 per image — the stand-in ``test_prompt_budget.py`` uses), 3 history turns, 3 images
    (the current frame plus the two history frames young enough to stay pixels under
    ``image_age_turns`` 2), 3 memory hits, one transcript. That is the shape of a real tutoring
    turn and it must fit inside 14 336 with the block present and nothing trimmed.
    """
    b = builder()
    turns = history(3)
    measure = FakeEngineTokenizer()
    result = b.fit(
        history=turns,
        frame=frame("c"),
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=hits(0.9, 0.5, 0.2),
        resolve=resolver(turns),
        measure=measure,
    )

    assert ui_block() in result.prompt.messages[0].text()
    assert result.fits is True, (
        f"a representative turn no longer fits: {result.tokens} > {MAX_PROMPT_TOKENS}"
    )
    assert dict(result.trimmed) == {"history_turns": 0, "images": 0, "memory_hits": 0}, (
        f"nothing should have been trimmed; got {dict(result.trimmed)}"
    )
    assert result.tokens == measure.count(result.prompt), "FitResult.tokens is measure(prompt)"


def test_fit_never_spends_the_block():
    """Adversarial, criterion 6: at a budget of 1 token the block is still there.

    The slip this is aimed at: making the block a fourth rung ("it is only instructions"), or
    moving it into the query-dependent suffix "to save the prefix" — either way the model loses
    the A2UI channel exactly on the long turns where a surface is most useful, and the turn log
    shows ``ui_valid: null`` again with no clue why.
    """
    b = builder(Budget(max_prompt_tokens=1))
    turns = history(3)
    result = b.fit(
        history=turns,
        frame=frame("c"),
        assists=AssistFlags(),
        transcript=TRANSCRIPT,
        memory_hits=hits(0.9, 0.5, 0.2),
        resolve=resolver(turns),
        measure=FakeEngineTokenizer(),
    )
    block = ui_block()

    assert result.fits is False
    assert block in result.prompt.messages[0].text(), (
        "the block was trimmed out of the system message; the system message is the floor "
        "(T-0020 ruling A1), not a rung"
    )
    assert TRANSCRIPT in result.prompt.messages[-1].text(), "the user's turn is intact"
    assert set(result.trimmed) == {"history_turns", "images", "memory_hits"}, (
        f"the ladder gained a rung: {sorted(result.trimmed)}. The rungs are history pixels, "
        "memory hits and whole turns — nothing else may be spendable"
    )
    assert dict(result.trimmed) == {"history_turns": 3, "images": 2, "memory_hits": 3}, dict(
        result.trimmed
    )
    assert sum(block in m.text() for m in result.prompt.messages) == 1, (
        "the block must not be duplicated into the suffix when the prefix is trimmed"
    )


# --- T-0055 criterion 7: the cached prefix changes exactly once ---------------------------------


def test_the_system_message_is_its_four_documented_parts_and_nothing_else():
    """T-0055 criterion 7: one place in the prefix changes, and it changes exactly once.

    ``system_message()`` joins four things with newlines — the session's system prompt,
    ``CLOSE_THE_THOUGHT``, ``SPOKEN_STYLE``, ``UI_INSTRUCTIONS`` — and T-0055 is allowed to change
    exactly one of them. This row is what makes "exactly once" checkable: with the first three
    pinned as literals by ``test_the_spoken_style_and_close_the_thought_rules_survive`` above, the
    only degree of freedom left in the cached prefix is the block itself, and a fifth paragraph,
    a reordering or a stray separator goes red here.

    The expected text is **derived**, never a copy of the block's bytes. A pin that pasted the
    block would be asserting current behaviour captured from the implementation
    (``.claude/rules/tests.md``) and would have to be re-pasted on every wording change, which is
    how a pin stops meaning anything.
    """
    text = builder().system_message().text()
    expected = "\n".join([SYSTEM, pb_module.CLOSE_THE_THOUGHT, pb_module.SPOKEN_STYLE, ui_block()])
    assert text == expected, (
        "the system message is no longer exactly [system prompt, CLOSE_THE_THOUGHT, SPOKEN_STYLE, "
        "UI_INSTRUCTIONS] joined by newlines. Every byte of it is the cached prefix of every "
        "prompt the product sends: something was added, reordered or re-joined, and the prefix "
        "cache is invalidated by it once per engine process.\n"
        f"length: {len(text)} expected {len(expected)}\n"
        f"only in the message: {text.replace(expected, '')!r}"
    )
    assert text.endswith(ui_block()), (
        "the block is joined last (F4: prompt_builder belongs to T-0053, so the block cannot move "
        "and its own first lines are the only place a framing sentence can land)"
    )


# --- T-0055 criterion 11: no client-side recovery, anywhere -------------------------------------


def _module_sources() -> list[tuple[Path, str]]:
    return sorted(
        (path, path.read_text(encoding="utf-8")) for path in (REPO / "local_astra").rglob("*.py")
    )


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """``id()`` of every Constant that is a module/class/function docstring — i.e. prose."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def test_no_module_outside_the_client_reads_the_fence():
    """T-0055 criterion 11 — **the human's ruling, as a test.**

        "[This] is only solvable by prompting a priori for us (we don't want to extract such
        things from that channel ourselves): find a way to elicit the model to output it outside
        the thought channel."  — the human, 2026-09-16

    If you are here because this row went red while you were trying to make the flight-booking
    panel appear: the fence you found in a ``thought`` commit is **not** yours to harvest. The
    thinking channel is discarded by design (``.claude/rules/conductor.md``: thinking content is
    never stored or replayed; T-0052 criterion 14 keeps reasoning deltas classified ``thought``
    whatever the fence state), and a surface written there is deleted by the chat template's own
    ``strip_thinking`` macro before anything of ours sees it. The fix is the wording of the
    instruction block — ``local_astra/conductor/ui_instructions.py``, this task — and if the
    measured eval (criteria 13–14) says wording is not enough, the option that is *not* extraction
    is a channel-aware grammar at decode time (T-0055 ambiguity 5, the human decides, and only
    after criterion 13 has a number).

    What the scan asserts, over the source text of every module in ``local_astra/``:

    * the literal ``<a2ui`` may appear in **code** only in ``conductor/llm_client.py``. Prose —
      docstrings and comments — is exempt everywhere: it explains the fence, it cannot recover
      one, and ``contracts/llm.py`` legitimately documents the fence lifecycle;
    * the names ``UI_OPEN`` / ``UI_CLOSE`` may be referenced only by ``conductor/llm_client.py``,
      which defines them, and ``conductor/ui_instructions.py``, which must *interpolate* them into
      the block rather than retyping the fence (criterion 5: the markers are the client's
      constants **by identity**). A retyped ``<a2ui>`` in the prompt survives a rename of the
      client's fence and sends every surface into the ``answer`` channel — with T-0052 merged,
      into the spoken one.

    Read as text and parsed with ``ast``; nothing is imported and no attribute is discovered by
    name pattern (``.claude/rules/tests.md``: a helper that collected names by pattern and called
    them killed nine rows on 2026-09-16). It follows that no module outside the client can look
    for a fence in a ``thought`` commit: it has no way to spell one.
    """
    fence_literal = "<a2ui"
    literal_allowed = {Path("conductor/llm_client.py")}
    name_allowed = {Path("conductor/llm_client.py"), Path("conductor/ui_instructions.py")}

    literal_offenders: list[str] = []
    name_offenders: list[str] = []
    for path, source in _module_sources():
        rel = path.relative_to(REPO / "local_astra")
        tree = ast.parse(source, filename=str(path))
        prose = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in prose
                and fence_literal in node.value
                and rel not in literal_allowed
            ):
                at = node.value.index(fence_literal)
                literal_offenders.append(
                    f"{rel}:{node.lineno}: …{node.value[max(0, at - 40) : at + 40]!r}…"
                )
            elif isinstance(node, ast.Name) and node.id in {"UI_OPEN", "UI_CLOSE"}:
                if rel not in name_allowed:
                    name_offenders.append(f"{rel}:{node.lineno}: {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in {"UI_OPEN", "UI_CLOSE"}:
                if rel not in name_allowed:
                    name_offenders.append(f"{rel}:{node.lineno}: .{node.attr}")

    assert not name_offenders, (
        "criterion 11: a module outside the LLM client reaches for the fence constants:\n  "
        + "\n  ".join(name_offenders)
        + "\nThe human has ruled that the surface is elicited by prompting, never extracted: "
        "no module may read the text of a thought commit looking for a fence, and no thought "
        "segment is ever re-labelled ui."
    )
    assert not literal_offenders, (
        "criterion 11 + criterion 5: the fence is spelled out in code outside "
        "conductor/llm_client.py:\n  "
        + "\n  ".join(literal_offenders)
        + "\nIn ui_instructions.py this is the block retyping the client's fence instead of "
        "interpolating llm_client.UI_OPEN / UI_CLOSE. The example and the markers must be the "
        "client's constants by identity (criterion 5), so that renaming the fence cannot leave a "
        "prompt teaching the old one — a surface fenced with a marker the splitter does not know "
        "goes to the answer channel, and with T-0052 merged that channel is spoken aloud.\n"
        "Anywhere else it is client-side recovery, which the human ruled out on 2026-09-16: "
        '"we don\'t want to extract such things from that channel ourselves".'
    )


def test_the_fence_scan_sees_the_files_it_claims_to_scan():
    """Criterion 11's too-wide half: the scan is worthless if it walks an empty tree.

    A source scan fails open in two silent ways — a wrong root (zero files, green forever) and an
    exemption so broad that the thing it guards is always exempt. So: the walk must find the
    package's modules, it must find the client among them, and the client's *own* fence definition
    must be visible to it (if ``llm_client.py`` were not on the allow-list, it would be the first
    offender). Nothing here asserts the guard's verdict; it asserts the guard can see.
    """
    sources = _module_sources()
    names = {path.relative_to(REPO / "local_astra") for path, _ in sources}
    assert len(sources) >= 20, f"the scan found only {len(sources)} modules under local_astra/"
    assert Path("conductor/llm_client.py") in names and Path("conductor/session.py") in names

    client = next(s for p, s in sources if p.name == "llm_client.py")
    tree = ast.parse(client)
    prose = _docstring_nodes(tree)
    defined_in_code = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in prose
        and "<a2ui" in node.value
    ]
    assert defined_in_code, (
        "the scan cannot see the fence literal in llm_client.py, where it is defined "
        "(UI_OPEN, UI_CLOSE): it would never see one anywhere else either"
    )
