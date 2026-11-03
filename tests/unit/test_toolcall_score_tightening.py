"""T-0086 criterion 4: the new ``score.exact`` only tightens the 4c7dfdc rule.

- **Tightening (Hypothesis):** for every call and ground truth, new ``exact`` implies the frozen 4c7dfdc ``exact``
  (``tests/unit/_frozen_score_4c7dfdc.py``), and the two are equal for every call that is not a google_search call
  against a google_search case. Total: neither raises.
- **Both directions on generated queries**, so the two named mutants die:
  - *control character checked before normalisation* (too wide): clean queries whose words are joined by any
    whitespace -- including the Cc characters ``str`` treats as whitespace (tab, LF, CR, VT, FF, U+001C-U+001F,
    NEL) -- keep the old verdict exactly;
  - *markup only* (too narrow): a clean query plus one injected problem other than markup (a non-whitespace Cc/Cf
    character, a tool token, a narration clause, padding past 32 words) is never exact, although the keywords
    are all present.
- The frozen oracle is proven to be 4c7dfdc's code (AST equality against ``git show``).

Regime: CPU, pure; Hypothesis default profile of the repo, no network.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests._git_range import require_commits
from tests.unit import _frozen_score_4c7dfdc as old
from tests.unit._toolcall_support import REPO, score_mod

CASES = REPO / "tests" / "eval" / "toolcall" / "cases.jsonl"
SCORE_REL = "local_astra/toolcall/score.py"


def new_exact(c: Any, g: Any) -> bool:
    res = score_mod().exact(c, g)
    assert isinstance(res, bool), f"exact() must return a bool, got {res!r}"
    return res


def _query_problem_exists() -> None:
    assert callable(getattr(score_mod(), "query_problem", None)), (
        "T-0086 not implemented: local_astra.toolcall.score.query_problem is missing"
    )


# --------------------------------------------------------------------------- the oracle is 4c7dfdc's


def test_the_frozen_oracle_is_4c7dfdcs_exact() -> None:
    """The embedded copy is 4c7dfdc's ``exact`` and helpers, function by function (AST dump equality).

    Skips (never passes) in a clone without 4c7dfdc (T-0157 criterion 11)."""
    require_commits(old.FROZEN_COMMIT)
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{old.FROZEN_COMMIT}:{SCORE_REL}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    frozen_src = Path(old.__file__).read_text()

    def defs(text: str) -> dict[str, str]:
        return {
            n.name: ast.dump(n)
            for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef) and n.name in old.FROZEN_NAMES
        }

    want, got = defs(src), defs(frozen_src)
    assert sorted(want) == sorted(old.FROZEN_NAMES)
    assert got == want
    assert old._WS.pattern == r"\s+"


# --------------------------------------------------------------------------- generators

_V1 = ("google_search", "click_at", "type_at", "scroll", "press_key")


def _case_gts() -> list[dict[str, Any]]:
    return [
        json.loads(x)["gt"]
        for x in CASES.read_text().splitlines()
        if x.strip() and json.loads(x)["gt"]
    ]


_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-5, 1200),
    st.floats(allow_nan=True),
    st.text(max_size=12),
)
_values = st.recursive(
    _scalars,
    lambda inner: (
        st.lists(inner, max_size=4) | st.dictionaries(st.text(max_size=4), inner, max_size=3)
    ),
    max_leaves=6,
)
_query_text = st.one_of(
    st.text(max_size=80),
    st.lists(
        st.sampled_from(
            [
                "weather",
                "lisbon",
                "paris",
                " ",
                "\t",
                "\n",
                "\x85",
                "\x00",
                "​",
                "{",
                "`",
                "|",
                "call:",
                "google_search",
                "scroll",
                "I will search",
                "i’ll google",
                "let me use",
                "i will survive",
                "w",
            ]
        ),
        max_size=40,
    ).map("".join),
)


@st.composite
def _calls(draw: st.DrawFn) -> dict[str, Any]:
    name = draw(st.sampled_from([*_V1, "navigate", ""]))
    args: dict[str, Any] = draw(
        st.dictionaries(
            st.sampled_from(["x", "y", "text", "press_enter", "direction", "combo", "query", "zz"]),
            _values,
            max_size=5,
        )
    )
    if name == "google_search" and draw(st.booleans()):
        args["query"] = draw(_query_text)
    if name in ("click_at", "type_at") and draw(st.booleans()):
        args["x"], args["y"] = draw(st.integers(0, 999)), draw(st.integers(0, 999))
    if draw(st.integers(0, 9)) == 0:
        return {"name": name, "arguments": draw(_values)}
    return {"name": name, "arguments": args}


@st.composite
def _gts(draw: st.DrawFn) -> dict[str, Any]:
    if draw(st.booleans()):
        return dict(draw(st.sampled_from(_case_gts())))
    name = draw(st.sampled_from(_V1))
    return {
        "name": name,
        "keywords": draw(
            st.lists(st.sampled_from(["weather", "lisbon", "i will", "W", "café"]), max_size=3)
        ),
        "bbox": draw(st.lists(st.integers(0, 999), min_size=4, max_size=4)),
        "text": draw(st.text(max_size=6)),
        "press_enter": draw(st.booleans()),
        "direction": draw(st.sampled_from(["up", "down"])),
        "combo": draw(st.sampled_from(["Ctrl+C", "Enter"])),
    }


@settings(max_examples=600, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(c=_calls(), g=_gts())
def test_new_exact_implies_the_4c7dfdc_exact_and_equals_it_off_google_search(
    c: dict[str, Any], g: dict[str, Any]
) -> None:
    """Criterion 4 (and criterion 1's "every other tool and branch is unchanged; malformed scores False and never
    raises"): new => old for every input; new == old unless it is a google_search call against a google_search
    case."""
    _query_problem_exists()
    n, o = new_exact(c, g), old.exact(c, g)
    assert (not n) or o, ("tightening violated", c, g)
    if not (c.get("name") == "google_search" and g.get("name") == "google_search"):
        assert n == o, ("a non-google_search verdict changed", c, g)


# --------------------------------------------------------------------------- both directions on queries

_CLEAN_WORDS = (
    "weather",
    "lisbon",
    "café",
    "東京",
    "c++",
    "site:github.com",
    "easyjet's",
    "near",
    "me",
    "how",
    "do",
    "i",
    "survive",
    "need",
    "a",
    "hero",
    "lyrics",
    '"quoted"',
    "tuesdays?",
    "2026",
    "scroll",  # ruling S5: an ordinary word, never tool syntax
)
_SEPARATORS = (
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x1f",
    "\x85",
    " ",
    " ",
    "　",
)


@st.composite
def _clean(draw: st.DrawFn) -> tuple[list[str], list[str]]:
    """``(words, separators)``: 1-30 clean words (no markup, no tool name, no clause+verb pair) and the whitespace
    between them, plus leading and trailing whitespace."""
    words = draw(st.lists(st.sampled_from(_CLEAN_WORDS), min_size=1, max_size=30))
    seps = draw(
        st.lists(st.sampled_from(_SEPARATORS), min_size=len(words) + 1, max_size=len(words) + 1)
    )
    return words, seps


def _join(words: list[str], seps: list[str]) -> str:
    out = seps[0]
    for w, s in zip(words, seps[1:], strict=True):
        out += w + s
    return out


@settings(max_examples=400, deadline=None)
@given(clean=_clean(), data=st.data())
def test_clean_queries_keep_the_old_verdict(
    clean: tuple[list[str], list[str]], data: st.DataObject
) -> None:
    """Criterion 4 (too wide; kills the control-character-before-normalisation mutant): a clean query of at most
    30 words, joined by any whitespace including whitespace Cc characters, is exact exactly when the 4c7dfdc rule
    says so -- for keywords drawn from its own words (present) and for a missing keyword."""
    _query_problem_exists()
    words, seps = clean
    q = _join(words, seps)
    kws = data.draw(st.lists(st.sampled_from(words), min_size=1, max_size=3))
    g = {"name": "google_search", "keywords": kws}
    c = {"name": "google_search", "arguments": {"query": q}}
    assert old.exact(c, g) is True, "fixture: the keywords are the query's own words"
    assert new_exact(c, g) is True, repr(q)
    g_missing = {"name": "google_search", "keywords": [*kws, "zermatt"]}
    assert new_exact(c, g_missing) is old.exact(c, g_missing) is False


_INJECTIONS = (
    "\x00",
    "\x07",
    "\x1b",
    "\x7f",
    "\x9b",
    "​",
    "‎",
    "⁠",
    "﻿",
    "­",
    "call:",
    "google_search",
    "click_at",
    "type_at",
    "press_key",
    "I will search",
    "i’ll google",
    "let me look up",
    "I'm going to use",
    "i need to query",
    "I AM GOING TO CALL",
)


@settings(max_examples=400, deadline=None)
@given(clean=_clean(), data=st.data())
def test_one_non_markup_problem_is_never_exact(
    clean: tuple[list[str], list[str]], data: st.DataObject
) -> None:
    """Criterion 1 (too narrow; kills the markup-only mutant): a clean query with one injected non-markup problem
    -- a non-whitespace Cc/Cf character, a tool token, a narration clause + verb, or padding past 32 words -- is
    not exact, although every keyword is present (the 4c7dfdc rule says exact)."""
    _query_problem_exists()
    words, seps = clean
    kws = data.draw(st.lists(st.sampled_from(words), min_size=1, max_size=3))
    if data.draw(st.booleans()):
        inj = data.draw(st.sampled_from(_INJECTIONS))
        at = data.draw(st.integers(0, len(words)))
        words2 = [*words[:at], inj, *words[at:]]
        seps2 = [*seps, " "]
    else:
        pad = ["pad"] * (33 - len(words) + data.draw(st.integers(0, 5)))
        words2 = [*words, *pad]
        seps2 = [*seps, *([" "] * len(pad))]
    q = _join(words2, seps2)
    c = {"name": "google_search", "arguments": {"query": q}}
    g = {"name": "google_search", "keywords": kws}
    assert old.exact(c, g) is True, "fixture: every keyword is present"
    assert new_exact(c, g) is False, repr(q)


@pytest.mark.parametrize(
    "q",
    ["weather\x85lisbon", "weather\x1clisbon", "\tweather lisbon\r\n"],
    ids=["nel", "file-sep", "tab-crlf"],
)
def test_named_row_against_the_control_before_normalisation_mutant(q: str) -> None:
    """The mutant's discriminating rows, named: each holds a Cc character that is whitespace."""
    _query_problem_exists()
    c = {"name": "google_search", "arguments": {"query": q}}
    assert new_exact(c, {"name": "google_search", "keywords": ["weather", "lisbon"]}) is True
