"""T-0088a criterion 1: the frozen special-token and stop tables, and ``drop_special_literals``.

The expected tables come from the subset fixture (the pinned file's ``added_tokens`` copied verbatim; the cd row
``tests/cd/unit/test_host_detok_parity.py`` proves the fixture and the tables against the pinned file) and from the
pinned ``generation_config.json`` (``eos_token_id: [1, 106, 50]``, read 2026-10-12).

Rows: the two tables; ``STOP_IDS``; ``drop_special_literals`` on each literal, on splices, and identity on text
with no literal (including ``a < b``, ``x |> y``, ``<|`` in prose, and the non-Gemma ``<|pad|>``/``<end_of_turn>``).
**Adversarial**: a single-pass deletion fails the splice rows; a too-wide deletion fails the identity rows; a
Hypothesis property checks the result has no literal, is a subsequence of the input and is a fixed point.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.unit._detok_support import SPECIAL_LITERALS, SPECIALS_BY_ID, fixture, special_mod


def test_the_hardcoded_copy_of_the_specials_matches_the_fixture() -> None:
    """Self-check: the test's own table equals the fixture's ``special: true`` added tokens, in id order."""
    added = sorted(
        (a for a in fixture()["tokenizer"]["added_tokens"] if a["special"]), key=lambda a: a["id"]
    )
    assert tuple((a["id"], a["content"]) for a in added) == SPECIALS_BY_ID
    assert len(SPECIALS_BY_ID) == 24


def test_special_tokens_is_the_frozen_tuple_of_the_24_literals_in_id_order() -> None:
    mod = special_mod()
    assert isinstance(mod.SPECIAL_TOKENS, tuple)
    assert mod.SPECIAL_TOKENS == SPECIAL_LITERALS


def test_special_ids_maps_each_id_to_its_literal() -> None:
    mod = special_mod()
    assert dict(mod.SPECIAL_IDS) == dict(SPECIALS_BY_ID)


def test_stop_ids_is_the_pinned_eos_token_id_set() -> None:
    mod = special_mod()
    assert frozenset({1, 106, 50}) == mod.STOP_IDS
    assert isinstance(mod.STOP_IDS, frozenset)


def test_stop_ids_are_special_ids() -> None:
    mod = special_mod()
    assert set(mod.SPECIAL_IDS) > mod.STOP_IDS
    assert [mod.SPECIAL_IDS[i] for i in sorted(mod.STOP_IDS)] == [
        "<eos>",
        "<|tool_response>",
        "<turn|>",
    ]


@pytest.mark.parametrize("lit", SPECIAL_LITERALS)
def test_drop_special_literals_deletes_each_literal(lit: str) -> None:
    drop = special_mod().drop_special_literals
    assert drop(f"I will{lit} click it.") == "I will click it."
    assert drop(lit) == ""
    assert drop(lit + lit) == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<|chan<channel|>nel>", ""),  # the task's splice: deleting <channel|> makes <|channel>
        ("x<tu<pad>rn|>y", "xy"),
        ("<<eos>|tool_ca<pad>ll>", ""),  # two deletions make <|tool_call>
        ('a<|tool_<|"|>call>b', "ab"),
        ("<|tu<|tu<eos>rn>rn>", ""),  # nested: one splice makes the next
        ("<|ch<eos>annel", "<|channel"),  # spliced, but not a literal: kept
    ],
)
def test_drop_special_literals_repeats_until_no_literal_is_left(text: str, expected: str) -> None:
    """Adversarial (single pass): one deletion can splice a new literal (the ``router._drop_channel_markers`` rule)."""
    assert special_mod().drop_special_literals(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain prose. With two sentences!",
        "is a < b and c > d?",
        "x |> y pipes x into y",
        "type <| then a name",
        "<|pad|> and <end_of_turn> and <|endoftext|> are not Gemma 4 specials",
        "<tool> <|tool_call <tool_call| channel|> |channel>",
        '<0x41> <|"> <"|>',
    ],
)
def test_drop_special_literals_is_the_identity_on_text_without_a_literal(text: str) -> None:
    """Adversarial (too wide): nothing but an exact literal is deleted."""
    assert special_mod().drop_special_literals(text) == text


_ALPHABET = [
    *SPECIAL_LITERALS,
    "<|",
    "|>",
    "<",
    ">",
    "|",
    "<|to",
    "ol_call>",
    "chan",
    "nel>",
    "a",
    " ",
    ".",
]


def _is_subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(ch in it for ch in small)


@settings(max_examples=400, deadline=None)
@given(st.lists(st.sampled_from(_ALPHABET), max_size=30))
def test_drop_special_literals_leaves_no_literal_and_only_deletes(parts: list[str]) -> None:
    text = "".join(parts)
    out = special_mod().drop_special_literals(text)
    assert not any(lit in out for lit in SPECIAL_LITERALS), (text, out)
    assert _is_subsequence(out, text), "only deletions are allowed"
    assert special_mod().drop_special_literals(out) == out
    if not any(lit in text for lit in SPECIAL_LITERALS):
        assert out == text
