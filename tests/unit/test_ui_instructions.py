"""T-0044 + T-0055: the A2UI instruction block itself.

T-0044 criteria 2, 3, 4, 5a (the example, the catalog, the fence, the size) and **T-0055
criteria 1, 2, 3, 4, 5, 10** — *where* the block tells the model to put the surface.

Why T-0055 exists (read this before changing a spelling list below)
==================================================================
On 2026-09-16, twice, the model built a complete, correctly fenced, root-bearing surface of 15
components **entirely inside its reasoning channel** (`logs/sessions/20260916-181929-3900425.jsonl`,
`logs/sessions/20260916-215207-2034852.jsonl` turn `t1`), then wrote — still inside the thought —
`I will say "I have created a flight booking interface for you."` and said exactly that aloud.
The thinking channel is discarded by design, so no `ui` segment was ever produced, no `ui.gate`
record exists in either log, and the human heard a confident claim while looking at an empty
panel. **The model obeyed every word it was given.** The prefix's only placement rule is about
speech ("Never leave the *answer* inside the thinking channel", in `prompt_builder`); the block
never says where the surface goes; and the sentence the model followed most literally — "Open and
close the fence in the same message" — it satisfied perfectly, inside the thought.

The block that failed is kept verbatim at `tests/fixtures/prompts/ui_block_t0044.txt` and is used
here as an **independent oracle**: every wording list this file adds must *fail* against that
text (`test_the_wording_lists_do_not_match_the_block_that_failed_twice`). A list that matched it
would be green on the block that produced two empty panels, which is the only way these rows can
lie.

The approach is ruled and is not open here: **prompting only**. No extraction from the thinking
channel, no client-side recovery, no promotion of a thought commit to a `ui` commit (the human,
2026-09-16). `tests/unit/test_prompt_ui_block.py` turns that ruling into a source scan.

The seam
========
``docs/tasks/T-0044.md`` names the file but not the symbols, so — as ``test_prompt_budget.py``
does for T-0020 — the names are fixed here and the implementer must match them::

    local_astra/conductor/ui_instructions.py

        UI_INSTRUCTIONS: str                   # the block joined into the cached system prefix
        UI_EXAMPLE: str                        # the one example payload, verbatim in the block
        ADVERTISED_COMPONENTS: frozenset[str]  # the component kinds the block names

Something wrong with a name? Say so to the reviewer; the implementer never edits ``tests/``.

Where the expected values come from
===================================
Not from the block. Every expectation here has an oracle outside the thing under test:

* the **example** is judged by ``local_astra.ui.A2UIValidator`` — the very object ``UiGate``
  holds (``ui_gate.UiGate.validator``), reading the vendored v0.9.1 schemas. A second copy of
  the schema, or an example checked by eye, is exactly the drift criterion 2 forbids;
* the **advertised set** is compared against ``A2UIValidator().component_kinds`` (18 basic kinds
  today) minus the three the HUD must not fetch and its own renderer renders as nothing (T-0164):
  the URL-fetching ``AudioPlayer``, ``Image`` and ``Video`` (CLAUDE.md: loopback only, T-0024). The 15 survivors are also written out by hand, so the day
  the ``edu`` catalog lands (T-0010 C4) this file goes red and the block is updated *in that
  task* instead of silently widening;
* the **fence** is ``llm_client.UI_OPEN`` / ``UI_CLOSE``, imported, never retyped;
* the **sizes** are 2 000 code points (criterion 5a, a structural change-detector — the token
  number is the engine's, criterion 5b) and 512 characters for the fenced example (criterion 4,
  derived below from the lean profile's canvas at a conservative 2 characters per token).

``ADVERTISED_COMPONENTS`` is not taken on trust either: every name in it must appear in the
block, so the frozenset cannot advertise one thing while the prose teaches another.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from local_astra.conductor.llm_client import UI_CLOSE, UI_OPEN
from local_astra.config import Profile
from local_astra.ui import A2UIValidator, parse_concatenated_json

try:  # the module is T-0044's to create; a missing import must not error at collection
    from local_astra.conductor import ui_instructions as _ui_instructions
except ImportError:  # pragma: no cover - until T-0044 lands
    _ui_instructions = None


def _symbol(name: str):
    """The block's public symbol, or the criterion's failure as ``NotImplementedError``."""
    if _ui_instructions is None:
        raise NotImplementedError(
            "local_astra/conductor/ui_instructions.py does not exist: the system prefix still "
            "never mentions the A2UI channel (docs/tasks/T-0044.md)"
        )
    value = getattr(_ui_instructions, name, None)
    if value is None:
        raise NotImplementedError(
            f"local_astra.conductor.ui_instructions defines no {name!r}; T-0044's seam is "
            "UI_INSTRUCTIONS (str), UI_EXAMPLE (str), ADVERTISED_COMPONENTS (frozenset[str])"
        )
    return value


def block() -> str:
    return _symbol("UI_INSTRUCTIONS")


def example() -> str:
    return _symbol("UI_EXAMPLE")


def advertised() -> frozenset[str]:
    value = _symbol("ADVERTISED_COMPONENTS")
    assert isinstance(value, frozenset), (
        "ADVERTISED_COMPONENTS is a frozenset[str] (it is compared against the validator's own "
        f"frozenset); got {type(value).__name__}"
    )
    return value


#: The whole vendored basic catalog, by hand (18 kinds, docs/12 §1.1 / T-0044 implementation
#: notes). Hand-written on purpose: it is the oracle the advertised set is derived from below.
BASIC_CATALOG_KINDS = frozenset(
    {
        "AudioPlayer",
        "Button",
        "Card",
        "CheckBox",
        "ChoicePicker",
        "Column",
        "DateTimeInput",
        "Divider",
        "Icon",
        "Image",
        "List",
        "Modal",
        "Row",
        "Slider",
        "Tabs",
        "Text",
        "TextField",
        "Video",
    }
)

#: Excluded from the block: ``AudioPlayer``, ``Image`` and ``Video`` fetch their media by URL, the
#: HUD asserts zero non-loopback requests (CLAUDE.md, T-0024), and the HUD's own renderer renders
#: all three as nothing (T-0164).
EXCLUDED_FROM_THE_HUD = frozenset({"AudioPlayer", "Image", "Video"})

#: Criterion 3's expected set: the basic catalog minus what the HUD cannot render (15 kinds).
EXPECTED_ADVERTISED = BASIC_CATALOG_KINDS - EXCLUDED_FROM_THE_HUD

#: The ``edu`` catalog T-0010 C4 has not built. Advertising one of these makes every surface that
#: uses it get pruned — the same ``ui_valid``-shaped failure with a different cause (criterion 3).
NOT_IN_THIS_CATALOG = ("Table", "Flashcard", "Math", "Chart", "Timer", "Steps", "AnnotatedImage")

#: Criterion 5a. A structural bound in *code points*, not a token claim: nothing in
#: ``local_astra/`` counts tokens (T-0005 amendment B10 deleted the character-based estimator),
#: so the token cost of the block is measured by the engine under ``/launch`` — criterion 5b's
#: ≤ 600 ``usage.prompt_tokens``, reported with its regime. This number only notices a block that
#: grew into a chapter.
MAX_BLOCK_CODE_POINTS = 2_000

#: Criterion 4's worst case: the fence must open *and* close inside one committed canvas
#: (``split_segments`` is called with ``in_ui=False`` per SSE chunk — llm_client B7/M7), so the
#: fenced example must be far shorter than a canvas. A canvas is 256 tokens in the lean profile;
#: at a deliberately pessimistic **2 characters per token** that is 512 characters. This is a
#: bound on the worst case, not a token count and not an estimate of the block's real cost.
CONSERVATIVE_CHARS_PER_TOKEN = 2

#: Any absolute URL in the block. The *one* legitimate URL-shaped string in an A2UI payload is
#: ``createSurface.catalogId``, which is an identifier compared as a string and never fetched
#: (``tests/unit/test_hud_no_audio_no_egress.py`` ruling A7: allowed in ``hud/src/ui/a2ui.js``
#: and nowhere else, with the behavioural guarantee of zero non-loopback requests). Everything
#: else is a fetch the HUD must never make.
URL_RE = re.compile(r"\b(?:https?|wss?)://[^\s\"'`,)>\\]+")


#: T-0055. The block that failed twice, byte for byte (``tests/fixtures/prompts/MANIFEST.md``:
#: commit d12ddca, 1 779 code points, sha256 2ec55d8d…). It is the oracle for "this wording list
#: is not vacuous": every list below must *fail* against it.
BASELINE_BLOCK_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "prompts" / ("ui_block_t0044.txt")
)

#: T-0055 criterion 6, for the failure messages only: the block was 1 779 of its 2 000 code
#: points before this task, so the placement rule has 221 to land in. Not asserted — the bound is
#: ``MAX_BLOCK_CODE_POINTS`` and the delta is reported by the implementer.
BLOCK_CODE_POINTS_BEFORE_T0055 = 1_779

#: T-0055 criterion 1. **Where the surface goes.** Every spelling is a phrase only a placement
#: rule would produce — never a bare word the rest of the block already contains (ruling A5 found
#: two vacuous lists in this very module by mutation). "one reply" is deliberately **absent**:
#: the block that failed says "…and </a2ui> in that one reply", so a list containing it would be
#: satisfied by the text that produced two empty panels.
PLACEMENT_SPELLINGS: tuple[str, ...] = (
    "in your reply",
    "in the reply",
    "in the reply itself",
    "in your answer",
    "in your spoken reply",
    "after you finish thinking",
    "after you have finished thinking",
    "once you have finished thinking",
    "once you are done thinking",
    "when you have finished thinking",
    "after your thinking ends",
    "after the thinking ends",
    "once your thinking has ended",
    "after you stop thinking",
    "when the thinking is over",
    "not while you are thinking",
    "not while thinking",
    "not in your thinking",
    "never in your thinking",
    "not inside your thinking",
    "never inside your thinking",
    "outside your thinking",
    "outside the thinking",
)

#: T-0055 criterion 2 / T-0044 criterion 4's fourth rule. **The surface is written once, whole.**
#: Anchored on completeness, never on "reply": "in your reply" belongs to
#: :data:`PLACEMENT_SPELLINGS` and the two lists must stay disjoint, or
#: ``test_the_placement_rule_sits_with_the_fence_rule`` degenerates into a restatement of
#: ``test_the_block_states_the_rules_of_the_channel``.
WRITE_ONCE_SPELLINGS: tuple[str, ...] = (
    "open and close",
    "opened and closed",
    "write it once",
    "write the whole surface",
    "the whole surface",
    "the complete surface",
    "complete surface",
    "all at once",
    "in one go",
    "in one piece",
    "one fence",
    "a single fence",
    "same message",
    "one message",
    "in full",
)

#: T-0055 criterion 1, second half, and finding F1. The chat template's own ``strip_thinking``
#: macro deletes the span ``<|channel>…<channel|>`` from every assistant message it re-renders,
#: so "what you write while thinking is thrown away" is a description of the machinery — the
#: strongest true sentence this block can say, and one it has never said. "never shown" alone is
#: **not** in this list: the block that failed already contains "A surface that is cut off is
#: never shown", about a different thing.
ERASURE_SPELLINGS: tuple[str, ...] = (
    "thrown away",
    "throw it away",
    "is erased",
    "erased",
    "is deleted",
    "deleted",
    "is discarded",
    "discarded",
    "is lost",
    "disappears",
    "vanishes",
    "is removed",
    "nobody sees",
    "no one sees",
    "nobody ever sees",
    "no one ever sees",
    "not shown to anyone",
    "shown to nobody",
    "shown to no one",
    "never reaches",
    "does not reach",
    "never exists",
    "does not exist",
    "is not kept",
    "never kept",
    "not kept",
)

#: T-0055 criterion 3, positive half: the block's opening ties the surface to *the reply*.
REPLY_FRAMING_SPELLINGS: tuple[str, ...] = (
    "your reply",
    "the reply",
    "one reply",
    "your answer",
    "what you say",
    "what you send",
    "you send",
    "the words you say",
)

#: T-0055 criterion 3, negative half. ``[ui channel]`` is the block's current header and "second
#: channel" its current framing; the model's own thought names the consequence back at us — *"I
#: should use the ``a2ui`` tool to create this UI"* — it read "a second channel" as a tool
#: callable from anywhere, including from inside a plan. A row that only looked for a new
#: sentence would pass with the old framing still sitting above it.
BANNED_FRAMINGS: tuple[str, ...] = ("second channel", "[ui channel]", "another channel")

#: T-0055 criterion 4 — **the adversarial list of the task.** Hand-copied added tokens of the
#: pinned tokenizer (``tokenizer.json`` of ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev
#: ``2ea8372``; chat template ``sha256:36e3a42e…`` per ``services/versions.lock``), with the
#: spellings this repo has recorded in docs/07 §Native tokens, docs/08 and
#: ``llm_client.THOUGHT_MARKERS``. Ids 98 / 100 / 101 / 105 / 106 are ``<|think|>``,
#: ``<|channel>``, ``<channel|>``, ``<|turn>``, ``<turn|>``.
SPECIAL_TOKENS: tuple[str, ...] = (
    "<|think|>",
    "<|/think|>",
    "<|channel>",
    "<channel|>",
    "<|turn>",
    "<turn|>",
    "<|image|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
    "<|endoftext|>",
    "<end_of_turn>",
    "<start_of_turn>",
    "<bos>",
    "<eos>",
    "<pad>",
)

#: The *shape* of an added token, so criterion 4 also catches the ones this list does not name
#: (24 exist; the file names 18). Nothing in an English instruction block is spelled ``<|…>`` or
#: ``<…|>``; the A2UI fence is ``<a2ui>``/``</a2ui>`` and carries no pipe.
SPECIAL_TOKEN_SHAPE = re.compile(r"<\|[^\s<>]{0,24}\|?>|<[^\s<>]{0,24}\|>")

#: T-0055 criterion 5: the example and the advertised order, frozen at the merge base (d12ddca)
#: as hand-written literals. Not read from the module — comparing the module against itself would
#: say nothing — and not regenerated: these bytes are what keeps T-0052's
#: ``tests/contract/test_prompt_example_reaches_a_surface.py`` valid while the prose around them
#: is rewritten.
FROZEN_UI_EXAMPLE = (
    '{"version":"v0.9.1","createSurface":{"surfaceId":"s","catalogId":'
    '"https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json"}}\n'
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":['
    '{"id":"root","component":"Column","children":["h","r1"]},'
    '{"id":"h","component":"Text","text":"Primes and squares","variant":"h3"},'
    '{"id":"r1","component":"Row","children":["a","b"]},'
    '{"id":"a","component":"Text","text":"2"},'
    '{"id":"b","component":"Text","text":"4"}]}}'
)

#: T-0055 criterion 5: the advertised component order, frozen. A tuple, not a set: this order is
#: part of the cached prefix bytes (a frozenset iterates in string-hash order — see
#: ``tests/unit/test_prompt_ui_block.py``'s two-``PYTHONHASHSEED`` row).
FROZEN_ADVERTISED_ORDER = (
    "Text",
    "Column",
    "Row",
    "List",
    "Card",
    "Divider",
    "Icon",
    "Tabs",
    "Modal",
    "Button",
    "CheckBox",
    "ChoicePicker",
    "TextField",
    "Slider",
    "DateTimeInput",
)

_BULLET = re.compile(r"^\s*[-*\u2022]\s+")
#: A sentence ends at ``. ! ?`` followed by whitespace and a capital. Deliberately conservative:
#: ``v0.9.1`` and ``catalog.json`` must not split, and neither must a hard-wrapped line.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def units(text: str) -> list[str]:
    """T-0055 criterion 2: the block split on **its own** line and bullet boundaries.

    A *unit* is one bullet (a line starting with ``-`` plus the hard-wrapped lines that continue
    it) or one sentence of a paragraph. Hard wraps are joined with a space first, because the
    block is wrapped at ~98 columns and a rule can therefore straddle a newline in the middle of
    a phrase ("say the\nanswer aloud").

    The criterion is "the same bullet **or** sentence", so two sentences of the same paragraph
    are two units: that is the point. The arrangement that failed twice had the placement rule in
    one place (``prompt_builder.CLOSE_THE_THOUGHT``) and the fence rule at the bottom of the
    block, and the model followed the local one.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                groups.append(current)
                current = []
            continue
        if _BULLET.match(line) and current:
            groups.append(current)
            current = []
        current.append(line.strip())
    if current:
        groups.append(current)

    out: list[str] = []
    for group in groups:
        joined = " ".join(group)
        if _BULLET.match(group[0]):
            out.append(joined)  # a bullet is one unit however many sentences it holds
        else:
            out.extend(_SENTENCE_END.split(joined))
    return [u.strip() for u in out if u.strip()]


def matches(text: str, spellings: tuple[str, ...]) -> list[str]:
    """Every spelling of ``spellings`` present in ``text``, case-folded."""
    low = text.lower()
    return [s for s in spellings if s in low]


def unit_stating_both(text: str) -> str | None:
    """The first unit of ``text`` that states **both** the placement rule and the write-once rule,
    or ``None``. Criterion 2's whole content, and the helper the adversarial sibling mutates.
    """
    for unit in units(text):
        if matches(unit, PLACEMENT_SPELLINGS) and matches(unit, WRITE_ONCE_SPELLINGS):
            return unit
    return None


def baseline_block() -> str:
    """The block as of T-0044, verbatim. Read as bytes-in-a-file, never regenerated."""
    return BASELINE_BLOCK_PATH.read_text(encoding="utf-8")


def opening(text: str) -> str:
    """The block's first two units. F4: ``prompt_builder`` joins the block **last** and belongs to
    T-0053, so the block's own first lines are the only place a framing sentence can land.
    """
    return " ".join(units(text)[:2])


def _example_components(payload: str) -> list[dict]:
    return [
        c
        for message in parse_concatenated_json(payload)
        if isinstance(message, dict) and isinstance(message.get("updateComponents"), dict)
        for c in message["updateComponents"]["components"]
    ]


# --- criterion 2: prompt and gate cannot drift -------------------------------------------------


def test_the_example_in_the_prompt_validates_through_the_gates_own_validator():
    """Criterion 2: the literal example is accepted by the validator ``UiGate`` holds.

    An example the gate would reject cannot ship: it would teach the model the one shape that
    produces ``ui_valid: false``. The oracle is ``A2UIValidator`` reading the vendored v0.9.1
    schemas, not a hand-check and not a second copy of the schema.
    """
    payload = example()
    report = A2UIValidator().validate_text(payload)

    assert report.ok, (
        "criterion 2: the example inside the block is not valid A2UI v0.9.1.\n"
        f"schema errors: {[str(e) for e in report.schema_errors]}\n"
        f"problems: {list(report.problems)}"
    )
    assert report.schema_errors == () and report.problems == ()
    assert report.topology is not None, (
        "the example must define components (a createSurface with no updateComponents teaches "
        "the model half a surface)"
    )
    assert report.topology.ok and report.topology.errors == (), (
        f"criterion 2: topology is not clean: {list(report.topology.errors)}"
    )

    ids = [c.get("id") for c in _example_components(payload)]
    assert "root" in ids, (
        f"the gate's topology step starts at 'root' (docs/12 §1.1); the example defines {ids}"
    )
    assert set(report.topology.reachable) == set(ids), (
        "every component of the example must be reachable from root in the *same* payload: a "
        "forward reference is exactly the shape that costs a resample (T-0010 A9), and an "
        f"orphan is a component the renderer ignores. reachable={list(report.topology.reachable)} "
        f"defined={ids}"
    )


def test_the_example_the_block_shows_is_the_example_that_was_validated():
    """Criterion 2, the join: ``UI_EXAMPLE`` appears verbatim inside ``UI_INSTRUCTIONS``.

    Without this the validator above would be judging a string the model never sees, and the
    block could drift from it in the next edit without a single test going red.
    """
    assert example() in block(), (
        "the payload the validator accepted does not appear verbatim in the block; the model is "
        "shown something else than what this suite checks"
    )


# --- criterion 3: only components that exist are advertised ------------------------------------


def test_the_catalog_this_task_teaches_is_the_basic_one():
    """Criterion 3's oracle, pinned (current behaviour — not pending).

    When the ``edu`` catalog lands (T-0010 C4: Math, Markdown, Table, Chart, Timer, Flashcard,
    AnnotatedImage, Steps) this row goes red *first* and says what to do: widen the block in that
    task. It is here so that "the advertised set is the catalog minus three" cannot quietly
    become true of a different catalog.
    """
    kinds = A2UIValidator().component_kinds
    assert kinds == BASIC_CATALOG_KINDS, (
        "the active catalog changed. T-0044 criterion 3: update ADVERTISED_COMPONENTS and the "
        f"block in the task that changes the catalog.\nadded: {sorted(kinds - BASIC_CATALOG_KINDS)}"
        f"\nremoved: {sorted(BASIC_CATALOG_KINDS - kinds)}"
    )
    for kind in NOT_IN_THIS_CATALOG:
        assert kind not in kinds, f"{kind} is in the catalog now — see the message above"


def test_every_component_the_block_offers_is_in_the_active_catalog_and_renderable():
    """Criterion 3: advertised ⊆ catalog, and exactly the 15 the HUD can render.

    **Negative half:** ``Table``, ``Flashcard``, ``Math`` and ``Chart`` are not advertised today —
    the catalog does not define them, so every surface that used one would be pruned: the same
    ``ui_valid``-shaped failure this task exists to fix, with a different cause.
    """
    names = advertised()
    kinds = A2UIValidator().component_kinds

    assert names <= kinds, (
        "criterion 3: the block advertises components the active catalog does not define "
        f"{sorted(names - kinds)} — every surface using one is pruned by the gate"
    )
    assert names == EXPECTED_ADVERTISED, (
        "criterion 3: the advertised set is the basic catalog minus AudioPlayer, Image and "
        "Video (they fetch by URL; the HUD asserts zero non-loopback requests and renders "
        f"them as nothing).\nunexpectedly advertised: {sorted(names - EXPECTED_ADVERTISED)}"
        f"\nmissing: {sorted(EXPECTED_ADVERTISED - names)}"
    )
    assert not (names & EXCLUDED_FROM_THE_HUD), sorted(names & EXCLUDED_FROM_THE_HUD)
    for kind in NOT_IN_THIS_CATALOG:
        assert kind not in names, (
            f"{kind} belongs to the edu catalog T-0010 C4 has not built; advertising it teaches "
            "the model a component the gate will prune"
        )


def test_the_block_names_every_component_it_advertises():
    """Criterion 3: the frozenset and the prose say the same thing.

    ``ADVERTISED_COMPONENTS`` is what the *tests* read; the model reads the block. If a name is
    only in the frozenset, this suite is checking a list nothing teaches.
    """
    text = block()
    missing = sorted(name for name in advertised() if name not in text)
    assert not missing, (
        f"these components are in ADVERTISED_COMPONENTS but nowhere in the block: {missing}"
    )


def test_the_example_uses_only_advertised_components():
    """Criterion 3 + loopback: the example may not demonstrate a kind the block does not offer.

    The example is the strongest instruction in the block — whatever it shows is what the model
    copies — so a media component there would both contradict the advertised set and invite the
    HUD to fetch a remote URL (CLAUDE.md: loopback only).
    """
    used = {c.get("component") for c in _example_components(example())}
    assert used <= advertised(), (
        f"the example uses components the block does not advertise: {sorted(used - advertised())}"
    )
    assert not (used & EXCLUDED_FROM_THE_HUD), sorted(used & EXCLUDED_FROM_THE_HUD)


def test_nothing_in_the_block_invites_a_remote_url():
    """Adversarial (implementation notes, loopback): the only URL allowed is the catalog id.

    The HUD renders whatever the surface says and the suite asserts zero non-loopback requests
    (CLAUDE.md). ``createSurface.catalogId`` is an identifier the renderer compares as a string
    and never fetches, so it is the single exemption; any other ``http(s)://`` or ``ws(s)://`` in
    the block — an example image, a "see the docs at …" — teaches the model to emit one on a
    real turn, and the model is the only thing between the block and the page.
    """
    allowed = set(A2UIValidator().known_catalog_ids)
    found = set(URL_RE.findall(block()))
    assert found <= allowed, (
        f"the block carries URLs that are not A2UI catalog identifiers: {sorted(found - allowed)}"
    )
    for component in _example_components(example()):
        rendered = json.dumps(component)
        assert not URL_RE.search(rendered), (
            f"a component of the example carries a URL; the HUD would fetch it: {component}"
        )


# --- criterion 4: the fence, verbatim ----------------------------------------------------------


def test_the_fence_markers_are_the_llm_clients_constants():
    """Criterion 4: the block's fence *is* ``llm_client.UI_OPEN`` / ``UI_CLOSE``.

    Asserted against the imported constants, never against a retyped ``<a2ui>``: if the client's
    fence is ever renamed, a hand-typed copy in the prompt would keep the tests green and send
    every surface into the ``answer`` channel.
    """
    text = block()
    assert UI_OPEN in text, f"the block never shows the opening fence {UI_OPEN!r}"
    assert UI_CLOSE in text, f"the block never shows the closing fence {UI_CLOSE!r}"
    assert text.count(UI_OPEN) == text.count(UI_CLOSE), (
        f"unbalanced fences in the block: {text.count(UI_OPEN)} open, {text.count(UI_CLOSE)} close"
    )

    strays = set(re.findall(r"</?a2ui[^>]*>", text)) - {UI_OPEN, UI_CLOSE}
    assert not strays, (
        f"the block shows fence-shaped markup the client does not know: {sorted(strays)} "
        f"(the only two markers are {UI_OPEN!r} and {UI_CLOSE!r})"
    )


def test_the_block_states_the_rules_of_the_channel():
    """Criterion 4: the four rules are *present*. A wording check, tolerant by construction.

    Each rule is looked for as any of several plain spellings, so the implementer is free in how
    it is phrased and only a **missing** rule goes red (ruling A1). Every spelling is therefore a
    *phrase that only the rule itself would produce*: a bare word is no evidence that a rule was
    stated, because the vocabulary of the rules is the vocabulary of the whole block. Two of these
    lists were vacuous in the first version of this file and the group-A review found it by
    mutation: ``"json"`` also occurs in the example's ``catalog.json`` and ``"aloud"`` in "say the
    answer aloud as usual", so deleting either bullet outright left this row green. The spellings
    below are anchored on the *restriction* ("json only", "never mention") rather than on the
    subject ("json", "aloud"), so a reworded bullet still passes and a deleted one fails.

    The rules and why each is in the block:

    * JSON only inside the fence — prose inside it fails the gate;
    * one surface per turn — the gate drops second surfaces (T-0010 A3/A7);
    * never name the fence in speech — CLAUDE.md's two channels; TTS would read the tag aloud;
    * the whole surface, opened and closed once — **reworded by T-0055.** This row used to say
      "inside one canvas", because ``split_segments`` was called with ``in_ui=False`` per chunk
      (llm_client B7/M7) and a straddling surface had its tail classified ``answer``. T-0052
      withdrew that: the client now carries the fence across canvases and reassembles it, so the
      block no longer promises "a surface that is cut off is never shown" — it would be false in
      the other direction. The rule that remains is that the surface is written once and whole.
      Where it is written is criterion 1's row, and the two must arrive in the same unit
      (criterion 2) because the model followed this rule, to the letter, inside its thought.

    The eval in ``tests/eval/test_a2ui_prompt_live.py`` is what says whether the wording *works*;
    this row only says the rule exists.
    """
    text = block().lower()
    rules = {
        # Anchored on the restriction, not on the word "json": the example's catalogId ends in
        # "catalog.json" and the fence is introduced as "one JSON object per line", so any
        # spelling that is satisfied by the bare word is satisfied with this bullet deleted.
        "json only inside the fence": (
            "json only",
            "only json",
            "nothing but json",
            "json and nothing else",
            "json, nothing else",
            "json - nothing else",
            "valid json only",
        ),
        "one surface per turn": ("one surface", "a single surface", "single surface", "one ui"),
        # "aloud" / "out loud" are how the block says *speak the answer* ("say the answer aloud as
        # usual"), so they are evidence of nothing; a prohibition is what this rule is.
        "never name the fence in speech": (
            "never mention",
            "do not mention",
            "don't mention",
            "never say",
            "do not say",
            "never speak",
            "do not speak",
            "never read",
            "do not read",
            "not in speech",
            "never in speech",
            "keep it out of speech",
        ),
        # T-0055: reworded. T-0052 withdrew the one-canvas rule in the client — a fence that
        # spans several canvases is now carried and reassembled — so "inside one canvas" is no
        # longer true and "A surface that is cut off is never shown" is no longer the reason.
        # What survives is: write the whole surface **once**, complete. The spellings are anchored
        # on completeness and never on "reply": "in your reply" belongs to PLACEMENT_SPELLINGS,
        # and the two lists must stay disjoint or test_the_placement_rule_sits_with_the_fence_rule
        # degenerates into a restatement of this row.
        "the whole surface, opened and closed once": WRITE_ONCE_SPELLINGS,
    }
    missing = [rule for rule, spellings in rules.items() if not any(s in text for s in spellings)]
    assert not missing, (
        f"criterion 4: the block does not state {missing}. Rewording is fine, but this row "
        f"recognises a finite list of phrasings, not meaning: if your wording states the rule and "
        f"is not below, add it here rather than bending the block. It looks for "
        f"{', '.join(sorted(s for r in missing for s in rules[r]))}"
    )


# --- criterion 5a and the one-canvas bound -----------------------------------------------------


def test_the_block_is_bounded_in_size():
    """Criterion 5a: ≤ 2 000 **code points**.

    Provenance of the number: it is a structural change-detector chosen in the task, not a token
    claim and not derived from one. Nothing in ``local_astra/`` counts tokens (T-0005 B10), so the
    real cost of the block — criterion 5b's ≤ 600 tokens — is measured by the engine under
    ``/launch`` as the difference in ``usage.prompt_tokens`` with and without the block, and
    reported with the full regime. A block over 2 000 characters is over budget by any reading;
    a block under it may still be over 600 tokens, and only the engine can say.
    """
    text = block()
    assert len(text) <= MAX_BLOCK_CODE_POINTS, (
        f"criterion 5a: the block is {len(text)} code points, over the {MAX_BLOCK_CODE_POINTS} "
        "bound. It is spent out of the 14 336-token prompt budget on every single turn"
    )


def test_the_example_cannot_straddle_a_canvas():
    """Adversarial, criterion 4: the fenced example fits one lean canvas with room to spare.

    ``split_segments`` is called with ``in_ui=False`` on every chunk (llm_client B7/M7) and one
    chunk is one committed canvas, so a surface that does not close inside one canvas has its
    tail classified ``answer`` — the failure mode this block is meant to prevent. If the
    *example* itself could not fit one canvas, the block would be teaching exactly that.

    The bound is on the worst case, not a token count: canvas = the lean profile's
    ``canvas_length`` (256 tokens) at a deliberately pessimistic **2 characters per token**
    (English JSON runs nearer 3-4), i.e. 512 characters for fence + payload + fence.
    """
    canvas_tokens = Profile(name="lean").diffusion.canvas_length
    limit = canvas_tokens * CONSERVATIVE_CHARS_PER_TOKEN
    fenced = UI_OPEN + example() + UI_CLOSE
    assert len(fenced) <= limit, (
        f"the fenced example is {len(fenced)} characters; a lean canvas is {canvas_tokens} tokens "
        f"and this bound is {limit} characters (2 chars/token, pessimistic). An example the model "
        "cannot copy inside one canvas teaches the straddle the gate never sees"
    )


# --- T-0055 criterion 1: the block says where the surface goes ----------------------------------


def test_the_block_says_where_the_surface_goes():
    """T-0055 criterion 1: a placement statement whose subject is the **surface**.

    The origin, in one line: the prefix's only placement rule is about speech ("Never leave the
    *answer* inside the thinking channel") and the block never says where the surface goes, so the
    model put a complete, correctly fenced, root-bearing surface inside its thought — twice — and
    then said aloud that it had created an interface.

    A wording pin over a list of spellings, in the shape ruling A1 settled: rewording is free,
    only a *missing* rule goes red. The mutation that must redden this row is "delete the
    placement sentence", and it is not hypothetical — the block that failed twice *is* that
    mutation, and ``test_the_wording_lists_do_not_match_the_block_that_failed_twice`` runs it on
    the fixture on every fast suite.

    Not asserted here (F2, criterion 4): the placement rule must name the boundary
    **behaviourally**. ``<channel|>`` is added token 101; pasting it into the block would put a
    control token inside the cached system turn.
    """
    text = block()
    found = matches(text, PLACEMENT_SPELLINGS)
    assert found, (
        "criterion 1: the block never says where the surface goes. Every rule it states is about "
        "what the surface must contain; none is about which part of the model's output carries "
        "it, and on 2026-09-16 the model twice wrote the whole fence inside its reasoning "
        "channel, where it is discarded, and said aloud that it had built an interface.\n"
        "This row recognises a finite list of phrasings, not meaning. If your sentence states "
        "the rule and is not below, add it to PLACEMENT_SPELLINGS rather than bending the block "
        f"(ruling A11's accepted trade-off). It looks for: {list(PLACEMENT_SPELLINGS)}\n"
        f"block ({len(text)} code points, was {BLOCK_CODE_POINTS_BEFORE_T0055} before this task):"
        f"\n{text}"
    )


def test_the_block_says_a_surface_written_while_thinking_is_thrown_away():
    """T-0055 criterion 1, second half, and finding F1 — the strongest true sentence available.

    The pinned chat template (``chat_template.jinja``, ``services/versions.lock``
    ``chat_template_revision = sha256:36e3a42e…``) carries a macro named ``strip_thinking`` and
    applies it to **every** assistant message it re-renders: it deletes the span
    ``<|channel>…<channel|>`` outright. A surface written inside the thought is therefore erased
    from the conversation *by the model's own machinery* — before our client's policy, before the
    session log, before anything we wrote. "What you write while you are thinking is thrown away;
    only what you write in your reply exists" is not a threat and not a house rule; it is a
    description of the mechanism, and the block has never contained it.

    The erasure statement must be **about thinking**: a unit that says something is discarded and
    a unit that mentions thinking, three lines apart, is not this sentence. ``never shown`` on its
    own is not in the list — the block that failed already says "A surface that is cut off is
    never shown", about a different thing entirely.
    """
    text = block()
    stated = [
        unit for unit in units(text) if matches(unit, ERASURE_SPELLINGS) and "think" in unit.lower()
    ]
    assert stated, (
        "criterion 1 / F1: the block never tells the model that what it writes while thinking is "
        "thrown away. This is literally true of the pinned chat template — strip_thinking deletes "
        "the thought span from every re-rendered assistant message — and it is the strongest "
        "elicitation this task has. One unit must say both 'while you are thinking' and that it "
        "is discarded; two units that each say half do not count, because the model follows the "
        "local instruction (that is the whole finding of 2026-09-16).\n"
        f"erasure spellings looked for: {list(ERASURE_SPELLINGS)}\n"
        f"units of the block:\n" + "\n".join(f"  - {u}" for u in units(text))
    )


# --- T-0055 criterion 2: the placement rule sits with the rule the model obeyed ------------------


def test_the_placement_rule_sits_with_the_fence_rule():
    """T-0055 criterion 2: one bullet or one sentence states **both**.

    The 21:52 evidence is that the model follows the most concrete, most *local* instruction about
    the fence. "Open and close the fence in the same message: begin ``<a2ui>`` only when you can
    finish the whole surface and ``</a2ui>`` in that one reply" is the sentence it obeyed to the
    letter — inside its thought. A placement rule three paragraphs away, in
    ``prompt_builder.CLOSE_THE_THOUGHT``, is precisely what it already ignored.

    So the two rules must arrive together. ``units()`` splits the block on its own line and bullet
    boundaries; one unit must match both lists. The lists are disjoint by construction (see
    ``PLACEMENT_SPELLINGS`` / ``WRITE_ONCE_SPELLINGS``) so that this row cannot be satisfied by
    either rule twice, and ``test_two_rules_three_lines_apart_are_not_one_unit`` proves the helper
    rejects the arrangement that failed.
    """
    text = block()
    unit = unit_stating_both(text)
    assert unit is not None, (
        "criterion 2: no single bullet or sentence of the block states both where the surface "
        "goes and that it is opened and closed once. That is today's arrangement — the fence rule "
        "at the bottom of the block, the placement rule (about speech) three paragraphs away in "
        "CLOSE_THE_THOUGHT — and it failed twice on 2026-09-16.\n"
        "Put them in one bullet (a line starting with '- ' plus its wrapped continuation lines) "
        "or in one sentence.\n"
        f"placement spellings present anywhere in the block: {matches(text, PLACEMENT_SPELLINGS)}\n"
        f"write-once spellings present anywhere in the block: "
        f"{matches(text, WRITE_ONCE_SPELLINGS)}\n"
        "units:\n" + "\n".join(f"  - {u}" for u in units(text))
    )
    assert matches(unit, PLACEMENT_SPELLINGS) and matches(unit, WRITE_ONCE_SPELLINGS)


def test_two_rules_three_lines_apart_are_not_one_unit():
    """Criterion 2's adversarial sibling: the **too-wide** half of the guard.

    A guard needs a row for being too wide as well as one for being absent
    (``.claude/rules/tests.md``). ``test_the_placement_rule_sits_with_the_fence_rule`` fires when
    the block is missing a rule; this one fires when ``units()`` is so coarse that "somewhere in
    the block" would satisfy it — which would make the row green on the exact arrangement that
    produced two empty panels.

    Two hand-written mutants, neither of them the real block: one that states the two rules in one
    bullet (must be accepted) and one that states them both, correctly, **three lines apart**
    (must be rejected). The second is today's shape with the placement rule added in the wrong
    place, i.e. the fix an implementer reaches for when they read criterion 1 without criterion 2.
    """
    together = (
        "Your reply has two parts: the words you say and, when it helps, one surface.\n"
        "- Write the whole surface once, in your reply, after you have finished thinking: open\n"
        "  <a2ui> and close </a2ui> there, never inside your thinking."
    )
    apart = (
        "Write the surface in your reply, after you have finished thinking.\n"
        "Components, and nothing else:\n"
        "Text, Column, Row, List, Card.\n"
        "There is no table component: a table is a Column of Rows of Text.\n"
        "- Between the fences: JSON only. No markdown, no backticks, no prose.\n"
        "- Open and close the fence in the same message: begin <a2ui> only when you can finish\n"
        "  the whole surface."
    )

    assert unit_stating_both(together) is not None, (
        "the helper rejects a block that puts both rules in one bullet — units() has become too "
        f"narrow to recognise the shape criterion 2 asks for.\nunits: {units(together)}"
    )
    assert matches(apart, PLACEMENT_SPELLINGS), "the mutant must really state the placement rule"
    assert matches(apart, WRITE_ONCE_SPELLINGS), "the mutant must really state the fence rule"
    assert unit_stating_both(apart) is None, (
        "criterion 2 is vacuous: units() accepts a block whose placement rule and fence rule are "
        "six lines apart — the arrangement of the prefix that failed twice on 2026-09-16. This "
        "row exists so that 'both rules are somewhere in the block' can never pass for 'both "
        f"rules are in the same unit'.\nunits: {units(apart)}"
    )


# --- T-0055 criterion 3: part of the reply, not a channel of its own ----------------------------


def test_the_surface_is_framed_as_part_of_the_reply():
    """T-0055 criterion 3: presence **and** absence.

    The recorded reason is in the model's own thought: *"I should use the ``a2ui`` tool to create
    this UI"*. It read "you have a second channel" as a tool callable from wherever it was
    standing — including from inside a plan. F3: the framing that fits what the model is already
    doing is *the reply has two parts — what you say, and, when it helps, one surface — and the
    reply begins when the thinking ends*.

    The absence half is not pedantry: a row that only looked for a new sentence would pass with
    "You have a second channel:" still sitting in the opening line, which is the sentence that
    produced the misreading.
    """
    text = block()

    present = [phrase for phrase in BANNED_FRAMINGS if phrase in text.lower()]
    assert not present, (
        f"criterion 3: the block still frames the surface as a channel of its own: {present}. "
        "The model read that as a tool it could call from inside its thinking, and it did. The "
        "surface is part of the one reply — alongside the spoken words — not a place the model "
        f"may write to whenever.\nblock:\n{text}"
    )

    head = opening(text)
    assert matches(head, REPLY_FRAMING_SPELLINGS), (
        "criterion 3: the block's opening sentence does not tie the surface to the reply. F4: "
        "prompt_builder joins the block **last** (and belongs to T-0053), so these first lines "
        "are the only place a framing sentence can land.\n"
        f"looked for {list(REPLY_FRAMING_SPELLINGS)} in the first two units:\n  {head}"
    )


# --- T-0055 criterion 4: no tokenizer special token, in any spelling ----------------------------


def test_the_block_contains_no_tokenizer_special_token():
    """T-0055 criterion 4 — **the adversarial row of the task**, and it is aimed at you.

    If you have just read finding F1 ("the thought is the span ``<|channel>…<channel|>``, so the
    boundary the model crosses is ``<channel|>``") the obvious next move is to paste that string
    into the block so the rule is concrete. **Do not.** ``<|think|>``, ``<|channel>``,
    ``<channel|>``, ``<|turn>``, ``<turn|>`` are *added tokens with* ``special: true`` in the
    pinned tokenizer (ids 98, 100, 101, 105, 106 of 24). Text in a chat message is encoded with
    added tokens matched, so the string does not reach the model as characters: it becomes control
    token 101 **inside the cached system turn**. Three consequences, none of them cosmetic —
    the ``gemma4`` reasoning parser sees a channel close that opens nothing; the prefix carries a
    control token in a text field, which is exactly what ``.claude/rules/conductor.md`` means by
    "the prefix must tokenise identically between prefill and the final request"; and the model is
    shown a control token in a position that invites it to echo one, which
    ``llm_client.THOUGHT_MARKERS`` catches for ``<|think|>`` and knows nothing about for
    ``<channel|>``.

    Name the boundary behaviourally instead ("once you have finished thinking", "in the reply the
    user receives") — that is what criterion 1's spelling list is made of. If you want the literal
    token, criterion 12's ``/tokenize`` run under ``/launch`` is the cheap way to find out what it
    becomes (post the candidate string, read the ids back); that is ambiguity 2, decided by the
    human, and until then this row forbids it.

    Provenance of the list: ``tokenizer.json`` of ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev
    ``2ea8372``, chat template ``sha256:36e3a42e…`` (``services/versions.lock``); the spellings are
    hand-copied from docs/07 §Native tokens, docs/08 and ``llm_client.THOUGHT_MARKERS``. The shape
    regex covers the added tokens this file does not name.
    """
    text = block()

    named = [token for token in SPECIAL_TOKENS if token in text]
    shaped = sorted(set(SPECIAL_TOKEN_SHAPE.findall(text)))
    assert not named and not shaped, (
        "criterion 4: the block carries a tokenizer special token. It will not reach the model as "
        f"characters — it is encoded as a control token inside the **cached system turn**.\n"
        f"named tokens found: {named}\nother added-token shapes found: {shaped}\n"
        "Name the thinking boundary behaviourally ('after you have finished thinking'), never by "
        "its token. See T-0055 F2 and criterion 12, which checks this again at the token layer."
    )


def test_the_special_token_scan_would_catch_the_token_an_implementer_reaches_for():
    """Criterion 4's too-wide half: prove the scan is not inert.

    The row above is a *guard*, and a guard that is absent and a guard that is trivially satisfied
    look identical when the block is clean. So: run the same detector over the block that the trap
    produces — today's text plus the one sentence F1 invites — and require it to fire. The mutants
    are the five spellings the task names by id, plus one the list does not name, to check the
    shape regex carries the rest of the 24.
    """
    for token in ("<|think|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|image|>"):
        mutant = baseline_block() + (
            f"\n- The thought ends at {token}; write the surface after it, in your reply."
        )
        named = [t for t in SPECIAL_TOKENS if t in mutant]
        shaped = SPECIAL_TOKEN_SHAPE.findall(mutant)
        assert named or shaped, (
            f"the special-token scan does not notice {token!r} pasted into the block; criterion 4 "
            "is inert and the trap F2 describes is wide open"
        )

    unseen = "<|separator|>"  # not in SPECIAL_TOKENS: the shape regex must carry it
    assert unseen not in SPECIAL_TOKENS
    assert SPECIAL_TOKEN_SHAPE.search(baseline_block() + unseen), (
        "the shape regex misses an added token this file does not name; 24 exist and "
        f"SPECIAL_TOKENS names {len(SPECIAL_TOKENS)}"
    )
    assert not SPECIAL_TOKEN_SHAPE.search(UI_OPEN + FROZEN_UI_EXAMPLE + UI_CLOSE), (
        "the shape regex matches the A2UI fence or the example payload — it would forbid the one "
        "markup the block is *required* to contain"
    )


# --- T-0055 criterion 5: the example and the catalog are frozen ---------------------------------


def test_the_example_and_the_catalog_did_not_move():
    """T-0055 criterion 5: ``UI_EXAMPLE`` and ``ADVERTISED_ORDER``, byte for byte.

    This task rewrites the prose around the example; the example, the advertised order and the
    fence markers are **frozen**. That is what keeps T-0052's
    ``tests/contract/test_prompt_example_reaches_a_surface.py`` — a module this task does not own —
    valid and unedited, and what keeps every T-0044 row above passing without a change.

    The expected values are hand-written literals taken at the merge base (d12ddca), not read back
    from the module: comparing the module against itself would assert nothing. Order matters as
    much as content — the tuple's order is part of the cached prefix bytes.
    """
    assert example() == FROZEN_UI_EXAMPLE, (
        "criterion 5: UI_EXAMPLE changed. It is the payload T-0052's contract test drives through "
        "the real UiGate and the one thing in the block the model copies literally; T-0055 is a "
        "prose change and must leave it alone (the leaf-property gap of T-0052 ruling A7 is a "
        "successor task, F6).\n"
        f"expected ({len(FROZEN_UI_EXAMPLE)} chars):\n{FROZEN_UI_EXAMPLE!r}\n"
        f"actual ({len(example())} chars):\n{example()!r}"
    )

    order = _symbol("ADVERTISED_ORDER")
    assert isinstance(order, tuple), (
        f"ADVERTISED_ORDER is a tuple — a set iterates in string-hash order and would give every "
        f"process a different cached prefix; got {type(order).__name__}"
    )
    assert order == FROZEN_ADVERTISED_ORDER, (
        "criterion 5: the advertised component order changed. The order is part of the cached "
        "prefix bytes and the list is frozen by this task.\n"
        f"expected: {list(FROZEN_ADVERTISED_ORDER)}\nactual:   {list(order)}"
    )
    assert ", ".join(FROZEN_ADVERTISED_ORDER) in block(), (
        "the block no longer lists the components in the frozen order; it is interpolated from "
        "ADVERTISED_ORDER and the join must survive the rewrite"
    )
    assert FROZEN_UI_EXAMPLE in block(), "the frozen example is no longer shown in the block"


# --- T-0055 criterion 10: the block is a literal, and importing it does nothing ------------------


def test_the_block_is_a_literal_and_importing_it_touches_nothing():
    """T-0055 criterion 10: no import-time I/O, and the block is not *generated*.

    Two slips this is aimed at, both named in T-0044's implementation notes and both invisible to
    every other row in this file. A block assembled at import time from ``A2UIValidator`` (or from
    any vendored schema) silently changes the cached system prefix whenever the schema changes; a
    block assembled from ``ADVERTISED_COMPONENTS`` — a ``frozenset`` — is re-ordered by string-hash
    randomisation, so every session would get a different prefix and the first prompt of every
    session would miss the prefix cache for a reason nobody would ever find in-process
    (``tests/unit/test_prompt_ui_block.py`` runs the two-``PYTHONHASHSEED`` half).

    Read as source, parsed with ``ast`` — never by importing and introspecting attributes by name
    pattern (``.claude/rules/tests.md``: discovering attributes by pattern and calling them killed
    nine rows on 2026-09-16).

    The import allow-list includes ``local_astra.conductor.llm_client`` on purpose: criterion 5
    says the fence markers in the block are ``UI_OPEN``/``UI_CLOSE`` **by identity, not retyped**,
    which means this module imports them.
    """
    source = (
        Path(__file__).resolve().parents[2] / "local_astra" / "conductor" / "ui_instructions.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    allowed_imports = {"__future__", "typing", "local_astra.conductor.llm_client"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported <= allowed_imports, (
        f"criterion 10: ui_instructions.py imports {sorted(imported - allowed_imports)}. The block "
        "is a module-level literal on the turn path; it reads no file, fetches no schema and "
        "builds nothing at import time. If the new import is provably free of I/O, add it to this "
        "allow-list in the same change and say why in the report."
    )

    io_names = {
        "open",
        "read_text",
        "read_bytes",
        "Path",
        "loads",
        "load",
        "A2UIValidator",
        "import_module",
        "get",
        "post",
    }
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert not (used & io_names), (
        f"criterion 10: ui_instructions.py touches {sorted(used & io_names)} — the block must be a "
        "literal. A generated block changes the cached system prefix whenever its source changes, "
        "and every session then pays a full prefill for a reason no in-process test can see."
    )

    assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            getattr(t, "id", None) == "UI_INSTRUCTIONS"
            for t in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        )
    ]
    assert len(assignments) == 1, (
        f"UI_INSTRUCTIONS must be assigned exactly once, at module level; found {len(assignments)}"
    )
    value = assignments[0].value
    assert isinstance(value, (ast.Constant, ast.JoinedStr)), (
        f"criterion 10: UI_INSTRUCTIONS is built by {type(value).__name__}, not written as a "
        "literal (a plain string or an f-string of literals)"
    )
    interpolated = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
    assert "ADVERTISED_COMPONENTS" not in interpolated, (
        "criterion 10: the block interpolates ADVERTISED_COMPONENTS, a frozenset. It iterates in "
        "string-hash order, so the cached prefix differs between processes — interpolate the "
        "ordered ADVERTISED_ORDER (T-0044 criterion 7's two-PYTHONHASHSEED row is the other half)"
    )


# --- T-0055: the wording lists above are not vacuous, proved against the block that failed ------


def test_the_wording_lists_do_not_match_the_block_that_failed_twice():
    """The anti-vacuity row for criteria 1, 2 and 3 — and the only one that cannot rot.

    ``.claude/rules/tests.md``: no asserting current behaviour, and a guard needs a row for being
    too wide. The deletion mutation for every wording row above is "remove the placement rule from
    the block", and that mutant exists as a file:
    ``tests/fixtures/prompts/ui_block_t0044.txt`` is the block the model was actually given on
    2026-09-16, when it twice built the whole surface inside its thought. If any list here matched
    that text, the corresponding row would have been green on the prefix that failed, and it would
    prove nothing about the new one.

    Read it in the other direction too: this row is the *mutation test of this module*, run on
    every fast suite instead of by hand at review time.
    """
    was = baseline_block()
    assert len(was) == BLOCK_CODE_POINTS_BEFORE_T0055, (
        f"the baseline fixture is {len(was)} code points, not {BLOCK_CODE_POINTS_BEFORE_T0055}: it "
        "has been edited. It is a historical record of commit d12ddca (see "
        "tests/fixtures/prompts/MANIFEST.md) and regenerating it makes both this row and the "
        "eval's baseline arm worthless"
    )

    assert not matches(was, PLACEMENT_SPELLINGS), (
        "PLACEMENT_SPELLINGS matches the block that failed twice "
        f"({matches(was, PLACEMENT_SPELLINGS)}); criterion 1's row would be green on the prefix "
        "that produced two empty panels. Anchor the spelling on the placement rule itself"
    )
    assert not matches(was, ERASURE_SPELLINGS), (
        f"ERASURE_SPELLINGS matches the old block ({matches(was, ERASURE_SPELLINGS)}) — most "
        "likely on 'A surface that is cut off is never shown', which is a rule about truncation, "
        "not about the thinking channel being discarded"
    )
    assert not matches(opening(was), REPLY_FRAMING_SPELLINGS), (
        f"REPLY_FRAMING_SPELLINGS matches the old opening ({matches(opening(was), REPLY_FRAMING_SPELLINGS)})"
    )
    assert unit_stating_both(was) is None, (
        "criterion 2's helper accepts the old block, which states the fence rule and no placement "
        "rule at all"
    )

    # …and the negative half really was true of it: this is the framing the model misread.
    assert [p for p in BANNED_FRAMINGS if p in was.lower()] == ["second channel", "[ui channel]"], (
        "the baseline fixture no longer carries the framing criterion 3 removes; either the "
        "fixture drifted or BANNED_FRAMINGS stopped describing it"
    )
    # the two lists criterion 2 depends on must stay disjoint, or that row restates another
    overlap = {a for a in PLACEMENT_SPELLINGS for b in WRITE_ONCE_SPELLINGS if a in b or b in a}
    assert not overlap, (
        f"PLACEMENT_SPELLINGS and WRITE_ONCE_SPELLINGS overlap on {sorted(overlap)}: criterion 2 "
        "would be satisfied by one rule stated once"
    )
