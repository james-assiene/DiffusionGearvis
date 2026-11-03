"""T-0172 A1 and A5: the announced-call rule ``local_astra.toolcall.announced_call.announces`` (docs/tasks/T-0172.md).

Every expected value is written out from the task file (A1's grammar and bounds, A5's table) or from the hand labels
in ``tests/fixtures/t0172/labels.jsonl`` (``build_labels.py``, made without importing the module), never from an
implementation. The module does not exist at BASE 3e4d45b7, so every new-behaviour row raises ``NotImplementedError``
from ``_mod()`` until it does (activated 2026-10-27).

**Readings this module pins** (named in the test-author's report):

- ``Announcement.verb`` is compared lower-cased; for "I will try to find ..." the verb is ``find`` (the optional
  ``try to`` is part of the grammar, the verb follows it), for "I will try one more time." and "I will try adding
  ..." it is ``try`` (``try`` is itself an action verb).
- ``Announcement.sentence`` is the sentence that matched: under ``tail`` the last one. It is compared stripped and
  without its final ``.``/``!``, since A1 does not say whether the terminator belongs to the sentence.
- The 240-character and 3-sentence bounds apply to the stripped text (A1: "stripped text non-empty, <= 240
  characters, <= 3 sentences").
- Not pinned: whether the intent must start its sentence (A1's closed lead-in list implies it; no row depends on
  it), and the scope of the conditional/``let me know`` checks beyond the sentence that matched (every row here puts
  them in that sentence).

Adversarial rows: ``test_deny_and_conditional`` (the task's named failure mode: a promise that is not an action, or
an action offered to the user), ``test_let_me_needs_an_action_verb``, ``test_tail_vs_all`` and the Hypothesis
properties. Mutants each is meant to catch (checked in a scratch copy, see the report): the conditional check
dropped; ``let me`` alone taken as intent; ``all`` used where ``tail`` is the default; no 240-character bound.

Regime: CPU, pure; deterministic (Hypothesis seeded); n = 1.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import gc
import importlib
import inspect
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from tests.unit._work_count import line_events

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.toolcall.announced_call"
LABELS = REPO / "tests" / "fixtures" / "t0172" / "labels.jsonl"
BUILDER = REPO / "tests" / "fixtures" / "t0172" / "build_labels.py"

SMOKE_1 = "I will search for the current weather in Paris again."
SMOKE_2 = "I will search for the latest stable version of Python."

#: A1's minimum lists, written out from the task file.
ACTION_MIN = frozenset(
    [
        "search",
        "look",
        "check",
        "find",
        "click",
        "tap",
        "press",
        "open",
        "go",
        "visit",
        "navigate",
        "browse",
        "scroll",
        "type",
        "enter",
        "fill",
        "select",
        "deselect",
        "add",
        "remove",
        "refresh",
        "reload",
        "submit",
        "fetch",
        "retry",
        "try",
    ]
)
DENY_MIN = frozenset(
    [
        "remember",
        "keep",
        "note",
        "wait",
        "tell",
        "explain",
        "answer",
        "provide",
        "know",
        "think",
        "be",
    ]
)


def _mod() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == MODULE:
            raise NotImplementedError(f"T-0172 A1: {MODULE} does not exist yet") from exc
        raise


def ann(text: str, mode: str | None = None) -> Any:
    m = _mod()
    return m.announces(text) if mode is None else m.announces(text, mode=mode)


def bare(sentence: str) -> str:
    return sentence.strip().rstrip(".!")


# --------------------------------------------------------------------------------------------- surface


def test_module_surface() -> None:
    """A1: ``announces(text, *, mode="tail")``, a frozen ``Announcement(verb, sentence)``, the two verb sets."""
    m = _mod()
    sig = inspect.signature(m.announces)
    params = list(sig.parameters.values())
    assert params[0].name == "text", sig
    assert sig.parameters["mode"].kind is inspect.Parameter.KEYWORD_ONLY, sig
    assert sig.parameters["mode"].default == "tail", "A1/H-T172-1: tail is the default"
    assert dataclasses.is_dataclass(m.Announcement), "Announcement is a dataclass"
    names = {f.name for f in dataclasses.fields(m.Announcement)}
    assert {"verb", "sentence"} <= names, names
    a = m.Announcement(**{f.name: "x" for f in dataclasses.fields(m.Announcement)})
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.verb = "y"
    assert isinstance(m.ACTION_VERBS, frozenset) and ACTION_MIN <= m.ACTION_VERBS, sorted(
        ACTION_MIN - set(m.ACTION_VERBS)
    )
    assert isinstance(m.DENY_VERBS, frozenset) and DENY_MIN <= m.DENY_VERBS, sorted(
        DENY_MIN - set(m.DENY_VERBS)
    )
    assert not (m.ACTION_VERBS & m.DENY_VERBS), sorted(m.ACTION_VERBS & m.DENY_VERBS)


def test_imports_stdlib_and_contracts_only() -> None:
    """A1: "stdlib and ``local_astra.contracts`` only"."""
    m = _mod()
    tree = ast.parse(Path(str(m.__file__)).read_text())
    bad = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import in {MODULE}"
            names = [node.module or ""]
        for n in names:
            top = n.split(".")[0]
            if n == "local_astra.contracts" or n.startswith("local_astra.contracts."):
                continue
            if top in sys.stdlib_module_names or top == "__future__":
                continue
            bad.append(n)
    assert not bad, f"{MODULE} imports outside stdlib and local_astra.contracts: {bad}"


# --------------------------------------------------------------------------------------------- A5


#: A5's ``None`` rows (both modes).
A5_NONE = [
    "I will remember that.",
    "I'll keep that in mind.",
    "Let me know if you need anything else.",
    "Should I search for that?",
    "I will search for it if you want.",
    "I can search for that.",
    "I searched for the weather.",
    "Let me think about that.",
    "I will be here.",
    "The Eiffel Tower is 330 metres tall.",
    "I will identify the cheapest item from the list.The cheapest item is the cotton tea towel at four dollars and"
    " seventy-five cents.",
]

#: A5's hits: (text, verb, the matching last sentence without its terminator, hit under ``all``).
A5_HITS = [
    (SMOKE_1, "search", "I will search for the current weather in Paris again", True),
    (SMOKE_2, "search", "I will search for the latest stable version of Python", True),
    ("I'll look that up.", "look", "I'll look that up", True),
    ("Let me check the weather in Paris.", "check", "Let me check the weather in Paris", True),
    (
        "Now I will scroll down to find the form.",
        "scroll",
        "Now I will scroll down to find the form",
        True,
    ),
    (
        "I will add the cotton tea towel to the cart.call:click_at{x:156,y:898}",
        "add",
        "I will add the cotton tea towel to the cart.call:click_at{x:156,y:898}",
        True,
    ),
    (
        "I don't see a booking form on this page. I will try to find the booking section.",
        "find",
        "I will try to find the booking section",
        False,
    ),
]


@pytest.mark.parametrize("mode", ["tail", "all"])
@pytest.mark.parametrize("text", A5_NONE)
def test_synthetic_table_none(text: str, mode: str) -> None:
    assert ann(text, mode) is None, f"A5: {text!r} must be None under {mode}"


@pytest.mark.parametrize(("text", "verb", "sentence", "under_all"), A5_HITS)
def test_synthetic_table_hits(text: str, verb: str, sentence: str, under_all: bool) -> None:
    got = ann(text)
    assert got is not None, f"A5: {text!r} is a hit (tail, the default)"
    assert got.verb.lower() == verb, (text, got)
    assert bare(got.sentence) == sentence, (text, got)
    assert ann(text, "tail") == got, "A1: the default mode is tail"
    got_all = ann(text, "all")
    if under_all:
        assert got_all is not None and got_all.verb.lower() == verb, (text, got_all)
    else:
        assert got_all is None, f"A5: the observation-plus-intent pair is None under all: {got_all}"


@pytest.mark.parametrize("text", [SMOKE_1, SMOKE_2])
@pytest.mark.parametrize("mode", ["tail", "all"])
def test_smoke_records_hit(text: str, mode: str) -> None:
    """The two T-0125 smoke records at b372eda (seq 904, 1047): hits with ``verb == "search"``."""
    got = ann(text, mode)
    assert got is not None and got.verb == "search", got


# --------------------------------------------------------------------------------------------- tail vs all

TAIL_ONLY = [
    "I don't see a booking form on this page. I will try to find the booking section.",
    "The cheapest item is the enamel mug at eleven dollars and fifty cents. I will add it to your cart.",
    "The page has loaded.\nI will scroll down to find the form.",
    "Actually, F14 is already selected. I will deselect it for you.",
]
BOTH = [
    "I will scroll down. I will click the booking button.",
    "Let me open the menu.\nI'll select the second option.",
]
NEITHER = [
    "I will scroll down. The page is long.",
    "I will check the cart.\nThe cart has two items.",
]


def test_tail_vs_all() -> None:
    """A1: ``tail`` reads the last sentence, ``all`` every sentence; ``tail`` is the default (H-T172-1)."""
    for text in TAIL_ONLY:
        assert ann(text) is not None, f"tail hit: {text!r}"
        assert ann(text, "tail") is not None, text
        assert ann(text, "all") is None, f"all needs every sentence: {text!r}"
    for text in BOTH:
        assert ann(text, "tail") is not None and ann(text, "all") is not None, text
    for text in NEITHER:
        assert ann(text, "tail") is None and ann(text, "all") is None, text
    two = ann("I will scroll down. I will click the booking button.")
    assert two.verb.lower() == "click" and bare(two.sentence) == "I will click the booking button"


# --------------------------------------------------------------------------------------------- bounds


def sized(n: int) -> str:
    """One intent sentence of exactly ``n`` characters (n >= 20)."""
    head = "I will search for "
    return head + "x" * (n - len(head) - 1) + "."


def test_bounds() -> None:
    """A1 size and question rules: 240 is in, 241 is out (on the stripped text); 3 sentences in, 4 out; any ``?``
    out; empty and whitespace out. Each ``None`` row has a hit twin, so an over-wide guard fails too."""
    assert len(sized(240)) == 240 and len(sized(241)) == 241
    assert ann(sized(240)) is not None, "240 characters is within the bound"
    assert ann(sized(241)) is None, "241 characters is over the bound"
    assert ann(sized(241), "all") is None
    assert ann("  \n" + sized(240) + " \n\t") is not None, "the bound applies to the stripped text"
    assert ann(sized(2000)) is None

    three = "The page loaded. The list is long. I will scroll down."
    four = "The page loaded. The list is long. Nothing yet. I will scroll down."
    assert ann(three) is not None and ann(four) is None
    assert ann("The page loaded\nThe list is long\nI will scroll down.") is not None
    assert ann("The page loaded\nThe list is long\nNothing yet\nI will scroll down.") is None, (
        "newlines split sentences"
    )
    assert ann("Done! Great! Fine. I will check the cart.") is None, "! splits sentences"
    assert ann("Done! Great! I will check the cart.") is not None
    assert ann("I will check the cart. I will add it. I will open it. I will pay.", "all") is None

    assert ann("I will search for the weather?") is None
    assert ann("Is it raining? I will search for the weather.") is None, "no ? anywhere"
    assert ann("I will search for the weather (or the forecast?) now.") is None
    assert ann("") is None
    assert ann("   \n\t  ") is None
    assert ann("", "all") is None


# --------------------------------------------------------------------------------------------- grammar


@pytest.mark.parametrize(
    ("text", "verb"),
    [
        ("I'll look that up.", "look"),
        ("I’ll look that up.", "look"),
        ("I'm going to click the button.", "click"),
        ("I’m going to click the button.", "click"),
        ("I am going to open the menu.", "open"),
        ("I shall press the submit button.", "press"),
        ("Let me check the cart.", "check"),
        ("let me check the cart.", "check"),
        ("I WILL SEARCH FOR THE WEATHER.", "search"),
        ("i will scroll down.", "scroll"),
    ],
)
def test_apostrophes(text: str, verb: str) -> None:
    """A1: every subject form, either apostrophe, case-insensitive."""
    got = ann(text)
    assert got is not None, text
    assert got.verb.lower() == verb, (text, got)


@pytest.mark.parametrize("lead", ["Now", "Next", "Actually", "Okay", "So", "Then", "First"])
def test_lead_ins(lead: str) -> None:
    for text in (f"{lead}, I will scroll down.", f"{lead} I will scroll down."):
        got = ann(text)
        assert got is not None and got.verb.lower() == "scroll", (text, got)


@pytest.mark.parametrize(
    ("text", "verb"),
    [
        ("I will now click the add to cart button.", "click"),
        ("I'll first open the menu.", "open"),
        ("I will then scroll down.", "scroll"),
        ("I will also add the mug.", "add"),
        ("Let me just check the cart.", "check"),
        ("I will try to open the menu.", "open"),
        ("I will try clicking the button again.", "try"),
        ("I will try one more time.", "try"),
        ("I will now try to find the form.", "find"),
        ("I'll retry the search.", "retry"),
        ("I will go back to the homepage.", "go"),
        ("I will type the address.", "type"),
        ("I will fill in the form.", "fill"),
        ("I will select the second seat.", "select"),
        ("I will deselect it for you.", "deselect"),
        ("I will remove the mug from the cart.", "remove"),
        ("I will refresh the page.", "refresh"),
        ("I will submit the form.", "submit"),
        ("I will navigate to the booking page.", "navigate"),
    ],
)
def test_adverbs_try_and_verbs(text: str, verb: str) -> None:
    got = ann(text)
    assert got is not None, text
    assert got.verb.lower() == verb, (text, got)


def test_deny_and_conditional() -> None:
    """Adversarial (the task's named failure mode): a promise that is no action, or an action offered to the user,
    never fires a re-ask. Each row has a hit twin differing only in the denied part, so a rule that refuses
    everything fails as well."""
    pairs = [
        ("I will search for it if you want.", "I will search for it."),
        ("I'll keep searching if you like.", "I'll search again."),
        ("Let me know what you find.", "Let me find it."),
        ("When you are ready, I will click submit.", "Then I will click submit."),
        ("I will click submit once you confirm.", "I will click submit."),
        ("I will search again should you need more.", "I will search again."),
        ("I will add it to the cart if you say so.", "I will add it to the cart."),
        ("I will wait for the page to load.", "I will scroll the page."),
        ("I will tell you the answer.", "I will check the answer."),
        ("I will explain the result.", "I will open the result."),
        ("I'll answer that.", "I'll look that up."),
        ("I will provide the details.", "I will find the details."),
        ("I will note that.", "I will check that."),
        ("I will think about it.", "I will look at it."),
        ("I will check the cart; let me know if it is wrong.", "I will check the cart."),
    ]
    for denied, twin in pairs:
        assert ann(twin) is not None, f"premise: {twin!r} is a hit"
        assert ann(denied) is None, f"A1 no denial: {denied!r}"
        assert ann(denied, "all") is None, f"A1 no denial (all): {denied!r}"
    assert ann("I will search for it. Let me know if that helps.") is None, (
        "the last sentence is 'let me know'"
    )


def test_let_me_needs_an_action_verb() -> None:
    """Adversarial: "let me" alone is not intent; the verb after it must be an action verb."""
    m = _mod()
    for verb in ("summarise", "say", "apologise", "recap"):
        assert verb not in m.ACTION_VERBS, f"{verb!r} is not an action on the screen or a tool"
        assert ann(f"Let me {verb} what I found.") is None, verb
        assert ann(f"I will {verb} what I found.") is None, verb
    assert ann("Let me be clear.") is None


# --------------------------------------------------------------------------------------------- review rows

#: The group A review's surviving mutants, each as ``(denied, hit twin)``: the twin differs only in the denied part,
#: so a guard that is too wide fails as well. Green at 62f571f1.
REVIEW_PAIRS = [
    # "try to" + a word that is no action (mutant: try-to non-action falls back to verb "try")
    ("I'll try to help.", "I'll try to open the menu."),
    ("I'll try to make it quick.", "I'll try to find it quickly."),
    # a condition on the user placed after the verb (mutant: "when" dropped from the conditional list)
    ("I will click submit when you are ready.", "I will click submit."),
    ("I will search again once you confirm.", "I will search again."),
    ("I will search for it if you like.", "I will search for it."),
    # "try" + a denied verb (mutant: the try+deny gate dropped, verb falls back to "try")
    ("I'll try remember.", "I'll try clicking."),
]


@pytest.mark.parametrize("mode", ["tail", "all"])
@pytest.mark.parametrize(("denied", "twin"), REVIEW_PAIRS, ids=[p[0] for p in REVIEW_PAIRS])
def test_review_rows_none(denied: str, twin: str, mode: str) -> None:
    assert ann(twin, mode) is not None, f"premise: {twin!r} is a hit under {mode}"
    assert ann(denied, mode) is None, f"A1/A1b: {denied!r} must be None under {mode}"


def test_anchored_intent() -> None:
    """A1b: the intent pattern starts the sentence (``match``, not ``search``); only the closed lead-ins precede it.
    Mutant: ``.match`` -> ``.search``."""
    assert ann("Actually, I will search again.") is not None, "premise: a closed lead-in"
    assert ann("Unless you object, I will search again.") is None
    assert ann("Unless you object, I will search again.", "all") is None
    assert ann("Hopefully I will find it.") is None
    assert ann("Then I will find it.") is not None


def test_conditional_scope_is_the_whole_text() -> None:
    """A1b: the conditional and ``let me know`` checks apply to every sentence, not only the matched one. Mutant:
    the check run on the last sentence only."""
    assert ann("For more, just ask. I will search again.") is not None, "premise: tail hit"
    assert ann("If you want more, just ask. I will search again.") is None
    assert ann("Let me know what you prefer. I will search again.") is None
    assert ann("When you are done, say so. I will scroll down.") is None


#: A1b's precision list (phrasal particles, deferrals, the user as object, a colon), each with a hit twin that
#: differs only in the denied part. These fail at 62f571f1 until A1b is implemented.
A1B_PAIRS = [
    ("I'll go with option two.", "I'll go to option two."),
    ("I'll look into it later.", "I'll look at it."),
    ("Let me check with you first.", "Let me check the cart first."),
    ("I'll go ahead and stop here.", "I'll go back and stop here."),
    ("I'll try my best.", "I'll try the search."),
    ("I'll check back later.", "I'll check the cart."),
    ("I will look forward to it.", "I will look for it."),
    ("Let me check: it is 20 degrees in Paris.", "Let me check the weather in Paris."),
    ("I'll go over the details.", "I'll go to the details page."),
    ("Let me go through the list.", "Let me go to the list."),
    ("I'll try harder.", "I'll try again."),
    ("I will search for flights tomorrow.", "I will search for flights now."),
    ("I'll check the cart next time.", "I'll check the cart this time."),
    ("I will refresh the page soon.", "I will refresh the page."),
    ("I'll look it up and ask you to confirm.", "I'll look it up and confirm."),
    # one row per remaining A1b item, so that deleting any single item reddens a row (review round 2)
    ("I'll look into it.", "I'll look at it."),
    ("I'll check back on the cart.", "I'll check the cart."),
    ("I'll check with the shop.", "I'll check the shop."),
    ("I'll search for it with you.", "I'll search for it."),
    ("I'll open it for you first.", "I'll open it."),
    ("I'll search for it later.", "I'll search for it."),
    ("I'll search again another time.", "I'll search again."),
]


@pytest.mark.parametrize("mode", ["tail", "all"])
@pytest.mark.parametrize(("denied", "twin"), A1B_PAIRS, ids=[p[0] for p in A1B_PAIRS])
def test_a1b_precision(denied: str, twin: str, mode: str) -> None:
    assert ann(twin, mode) is not None, f"premise: {twin!r} is a hit under {mode}"
    assert ann(denied, mode) is None, f"A1b: {denied!r} must be None under {mode}"


@pytest.mark.parametrize("mode", ["tail", "all"])
def test_particle_only_straight_after_its_verb(mode: str) -> None:
    """A1b: a particle refuses only when it follows its verb directly ("go ... with the form" is still ``go``)."""
    got = ann("I'll go to the page with the form.", mode)
    assert got is not None and got.verb.lower() == "go", got


@pytest.mark.parametrize("mode", ["tail", "all"])
@pytest.mark.parametrize(
    ("text", "verb"),
    [
        ("I'll go ahead and add the mug to your cart.", "add"),
        ("I will go ahead and click the submit button.", "click"),
    ],
)
def test_a1c_go_ahead_and_verb_hits(text: str, verb: str, mode: str) -> None:
    """A1c: ``go ahead and <verb>`` is judged as ``<verb>``."""
    got = ann(text, mode)
    assert got is not None, f"A1c: {text!r} is a hit under {mode}"
    assert got.verb.lower() == verb, (text, got)


@pytest.mark.parametrize("mode", ["tail", "all"])
@pytest.mark.parametrize("text", ["I'll go ahead and stop here.", "I'll go ahead."])
def test_a1c_go_ahead_stays_none(text: str, mode: str) -> None:
    """A1c: ``go ahead and stop`` is judged as ``stop`` (no action); ``go ahead`` alone stays ``None``."""
    assert ann(text, mode) is None, f"A1c: {text!r} must be None under {mode}"


# --------------------------------------------------------------------------------------------- corpus


def label_rows() -> list[dict[str, Any]]:
    lines = [json.loads(x) for x in LABELS.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines[1:]


def test_label_fixture_is_consistent() -> None:
    """The fixture agrees with its own provenance line (green now: it is data, not behaviour)."""
    lines = [json.loads(x) for x in LABELS.read_text(encoding="utf-8").splitlines() if x.strip()]
    prov = lines[0]["provenance"]
    rows = lines[1:]
    assert len(rows) == prov["candidates"] == 116
    assert sum(r["label"] for r in rows) == prov["positives"] == 108
    assert sum(not r["label"] for r in rows) == prov["negatives"] == 8
    assert not any(r["source"].split("/", 1)[0] == "sessions" for r in rows), (
        "no row from logs/sessions/ (private chats)"
    )
    smoke = sorted((r["request_id"], r["answer"]) for r in rows if r["smoke"])
    assert smoke == [("t2-gen-s7", SMOKE_1), ("t3-gen-s2", SMOKE_2)], smoke


def test_builder_never_imports_the_rule() -> None:
    """A4: "labels made without importing ``announced_call``": the builder imports the standard library only."""
    tree = ast.parse(BUILDER.read_text())
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    bad = [n for n in mods if n.split(".")[0] not in sys.stdlib_module_names | {"__future__"}]
    assert not bad, f"build_labels.py imports outside the standard library: {bad}"


def test_labelled_corpus_from_fixture() -> None:
    """A4 on the fixture alone (the eval re-reads the logs): ``tail`` hits no labelled negative, hits >= 95 % of
    the 108 positives and both smoke records; ``all`` hits no negative and is never wider than ``tail``."""
    rows = label_rows()
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    false_hits = [r["answer"][:80] for r in neg if ann(r["answer"]) is not None]
    assert not false_hits, f"A4 precision 1.0: hits on labelled negatives: {false_hits}"
    hits = [r for r in pos if ann(r["answer"]) is not None]
    misses = [r["answer"][:80] for r in pos if ann(r["answer"]) is None]
    assert len(hits) / len(pos) >= 0.95, f"A4 recall {len(hits)}/{len(pos)}; misses: {misses}"
    assert all(ann(r["answer"]) is not None for r in rows if r["smoke"]), "both smoke records hit"
    assert not [r for r in neg if ann(r["answer"], "all") is not None]
    wider = [r["answer"][:80] for r in rows if ann(r["answer"], "all") and not ann(r["answer"])]
    assert not wider, f"all hit where tail did not: {wider}"


# --------------------------------------------------------------------------------------------- properties

FRAGMENTS = [
    "I will",
    "I'll",
    "I’ll",
    "Let me",
    "let me know",
    "I am going to",
    "I'm going to",
    "I shall",
    "Now,",
    "Actually",
    "try to",
    "try",
    "just",
    "search",
    "click",
    "scroll",
    "add",
    "remember",
    "keep",
    "wait",
    "be",
    "the",
    "cart",
    "weather",
    "if you",
    "when you",
    "page",
    ".",
    "!",
    "?",
    "\n",
    ",",
    " ",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
]
TEXT = st.one_of(
    st.lists(st.sampled_from(FRAGMENTS), max_size=40).map(" ".join),
    st.text(max_size=300),
)


@seed(172001)
@settings(max_examples=400, deadline=None)
@given(text=TEXT, mode=st.sampled_from(["tail", "all"]))
def test_never_raises_and_respects_a1(text: str, mode: str) -> None:
    """A1 as properties: never raises; ``None`` for an empty stripped text, a ``?`` anywhere, over 240 stripped
    characters; a hit has an action verb that is not denied, a sentence found in the text, and no conditional
    addressed to the user nor ``let me know`` in that sentence; an ``all`` hit is a ``tail`` hit."""
    m = _mod()
    got = m.announces(text, mode=mode)
    stripped = text.strip()
    if not stripped or "?" in text or len(stripped) > 240:
        assert got is None, (text, got)
    if got is not None:
        assert isinstance(got, m.Announcement)
        assert got.verb.lower() in m.ACTION_VERBS and got.verb.lower() not in m.DENY_VERBS, got
        sentence = " ".join(bare(got.sentence).lower().split())
        assert sentence in " ".join(text.lower().split()), (text, got)
        assert not re.search(r"\b(?:if|when|once|should) you\b|\blet me know\b", sentence), (
            text,
            got,
        )
    if mode == "all" and got is not None:
        assert m.announces(text, mode="tail") is not None, text


# --------------------------------------------------------------------------------------------- linear

SHAPES: list[Callable[[int], str]] = [
    lambda n: ("I will " * (n // 7 + 1))[:n],
    lambda n: "a" * n,
    lambda n: ("I will search for the " + "x " * n)[:n],
    lambda n: (". " * n)[:n],
    lambda n: "\n" * n,
    lambda n: ("I'll try try " * n)[:n],
    lambda n: (" " * (n - 30)) + "I will search for the weather.",
    lambda n: ("Let me " + "let me " * n)[:n],
]


def test_linear_never_raises() -> None:
    """A1 "linear": traced Python line events at 100k characters are at most 20x those at 10k (linear predicts
    <= 10x, quadratic 100x; T-0129 rule), and ``time.thread_time`` at 10k characters is under 50 ms per call (min
    of 3, gc off). Every shape returns without raising."""
    m = _mod()
    root = str(m.__file__)
    slow: list[str] = []
    for i, shape in enumerate(SHAPES):
        for mode in ("tail", "all"):
            text, huge = shape(10_000), shape(100_000)
            small = line_events(functools.partial(m.announces, text, mode=mode), root)
            big = line_events(
                functools.partial(m.announces, huge, mode=mode), root, cap=max(20 * small, 2_000)
            )
            assert big <= max(20 * small, 2_000), (i, mode, small, big)
            best = float("inf")
            gc_was = gc.isenabled()
            gc.disable()
            try:
                for _ in range(3):
                    t0 = time.thread_time()
                    m.announces(text, mode=mode)
                    best = min(best, time.thread_time() - t0)
            finally:
                if gc_was:
                    gc.enable()
            if best > 0.05:
                slow.append(f"shape {i} {mode}: {best * 1000:.1f} ms")
    assert not slow, f"A1: over 50 ms CPU on 10k characters: {slow}"
