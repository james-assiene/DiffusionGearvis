"""T-0063 criterion 8 (structure), criterion 4's reserved-id rule on compiled automata, and the task's
adversarial rows.

Construction under test (diffgemma-fa SPEC §3.5-3.6, ``compile/automaton.py``):
``FA_total = HEADER · FA_grammar · STOP · Σ*`` with ``HEADER = 100 · Σ_name{1,8} · 107 · 101``, the stop set
taken from the pinned ``generation_config.json`` (``eos_token_id = [1, 106, 50]``), and ACC the only
accepting state with an unscored ``ACC --Σ--> ACC`` tail. Acceptance is decided by the independent walker
in ``tests/cd/_compile_support.py``.

Hand-written BFCL-dialect functions below keep these rows independent of the BFCL fixture; the rows that
name a real record (``live_simple_122-78-0``) need tests/fixtures/cd/bfcl/.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from tests.cd._compile_support import (
    CHANNEL_CLOSE,
    CHANNEL_OPEN,
    EOS,
    GENERATION_CONFIG,
    PAD,
    PHASE0_HEADER,
    RESERVED_IDS,
    STOP_IDS,
    TURN_CLOSE,
    TURN_OPEN,
    VOCAB_SIZE,
    bfcl_test_cut_ids,
    compiled_bfcl,
    compiled_function,
    compiled_sudoku,
    encode,
    sudoku_cut,
    vendored,
    walker,
    warm_bfcl,
    warm_sudoku,
)

#: Shaped like a BFCL-Live simple schema: one required integer, one optional string with a default.
LOOKUP_FN: dict[str, Any] = {
    "name": "lookup_user",
    "description": "Look up a user record by id.",
    "parameters": {
        "type": "dict",
        "required": ["user_id"],
        "properties": {
            "user_id": {"type": "integer", "description": "The user's id."},
            "special": {"type": "string", "description": "A free-text note.", "default": "none"},
        },
    },
}
LOOKUP_OK = '{"user_id": 7890, "special": "black"}'
LOOKUP_PRETTY = '```json\n{\n  "user_id": 7890,\n  "special": "black"\n}\n```'

#: BFCL's ``type: any`` (the live_simple_117-73-0 shape): one required property of any JSON type.
ANY_FN: dict[str, Any] = {
    "name": "echo_value",
    "description": "Echo a value of any type.",
    "parameters": {
        "type": "dict",
        "required": ["model"],
        "properties": {"model": {"type": "any", "description": "Any value."}},
    },
}


def _frame(body: str | list[int], *, stop: int = EOS) -> list[int]:
    ids = encode(body) if isinstance(body, str) else body
    return [*PHASE0_HEADER, *ids, stop]


def _stop_set_from_generation_config() -> frozenset[int]:
    cfg = json.loads(GENERATION_CONFIG.read_text())
    return frozenset(int(t) for t in cfg["eos_token_id"])


def _sample_automata() -> list[tuple[str, Any]]:
    """Automata every structural row runs on: a spread of Sudoku puzzles and the hand-written schemas."""
    cut = sudoku_cut("test")
    out = [(rid, compiled_sudoku(p)) for rid, p, _ in cut[::25]]
    out.append(("LOOKUP_FN", compiled_function(LOOKUP_FN)))
    return out


# --------------------------------------------------------------------------- criterion 8: structure


def test_the_generation_config_stop_set_is_the_phase0_stop_set() -> None:
    """Fixture check: the pinned generation_config agrees with diffgemma-fa Phase 0's END_TOKENS."""
    assert _stop_set_from_generation_config() == STOP_IDS == {1, 106, 50}


def test_every_sampled_automaton_passes_the_vendored_structural_invariants() -> None:
    check = vendored("automaton")._assert_structural_invariants
    for name, a in _sample_automata():
        check(a, name)


def test_every_sudoku_test_cut_automaton_passes_the_structural_invariants() -> None:
    check = vendored("automaton")._assert_structural_invariants
    warm_sudoku("test")
    for rid, puzzle, _ in sudoku_cut("test"):
        check(compiled_sudoku(puzzle), rid)


def test_every_bfcl_test_cut_automaton_passes_the_structural_invariants() -> None:
    check = vendored("automaton")._assert_structural_invariants
    warm_bfcl(bfcl_test_cut_ids())
    for rid in bfcl_test_cut_ids():
        check(compiled_bfcl(rid), rid)


def test_the_edges_into_acc_carry_exactly_the_generation_config_stop_set() -> None:
    want = _stop_set_from_generation_config()
    for name, a in _sample_automata():
        w = walker(a)
        acc = w.acc()
        into = [cls for src, dst, cls in w.edges if dst == acc and src != acc]
        assert into, f"{name}: nothing leads into ACC"
        for cls in into:
            assert w.class_size(cls) == len(want) and w.members_among(cls, want) == want, (
                f"{name}: an edge into ACC carries a class of size {w.class_size(cls)} "
                f"containing {sorted(w.members_among(cls, want))}; want exactly {sorted(want)}"
            )


@pytest.mark.parametrize("stop", sorted(STOP_IDS))
def test_each_stop_id_alone_ends_an_accepted_answer(stop: int) -> None:
    a = compiled_function(LOOKUP_FN)
    assert walker(a).accepts(_frame(LOOKUP_OK, stop=stop))


@pytest.mark.parametrize("not_a_stop", [PAD, 2, 3, TURN_OPEN, CHANNEL_OPEN, CHANNEL_CLOSE])
def test_a_non_stop_id_does_not_end_an_answer(not_a_stop: int) -> None:
    """Too-wide guard for the stop set: PAD, BOS, UNK, ``<|turn>`` and the channel ids are not stops."""
    a = compiled_function(LOOKUP_FN)
    assert not walker(a).accepts(_frame(LOOKUP_OK, stop=not_a_stop))


def test_acc_tail_is_the_whole_vocabulary() -> None:
    for name, a in _sample_automata():
        w = walker(a)
        acc = w.acc()
        out = [(dst, cls) for src, dst, cls in w.edges if src == acc]
        assert [d for d, _ in out] == [acc], f"{name}: ACC must only loop on itself, has {out}"
        assert w.class_size(out[0][1]) == VOCAB_SIZE, f"{name}: the ACC self-loop is not Σ"


def _tail_rows() -> list[tuple[str, list[int]]]:
    rng = random.Random(3_5)  # SPEC §3.5
    return [
        ("pad_run", [PAD] * 64),
        ("every_stop_id", [1, 106, 50, 1, 106, 50]),
        ("channel_and_turn_ids", [CHANNEL_OPEN, CHANNEL_CLOSE, TURN_OPEN, TURN_CLOSE]),
        ("control_and_top_ids", [0, 2, 3, 4, VOCAB_SIZE - 1]),
        ("random_ids", [rng.randrange(VOCAB_SIZE) for _ in range(200)]),
    ]


@pytest.mark.parametrize(("label", "tail"), _tail_rows(), ids=[r[0] for r in _tail_rows()])
@pytest.mark.parametrize("stop", sorted(STOP_IDS))
def test_after_acceptance_any_id_sequence_is_accepted(
    stop: int, label: str, tail: list[int]
) -> None:
    """SPEC §3.5 traps 3-4: the tail is Σ*, not PAD*, and every stop id enters it."""
    a = compiled_function(LOOKUP_FN)
    assert walker(a).accepts([*_frame(LOOKUP_OK, stop=stop), *tail]), label


# --------------------------------------------------------------------------- reserved ids (criterion 4)


def test_no_reserved_id_labels_a_grammar_edge() -> None:
    """Inside the grammar region (after the header's 101, before ACC) no edge carries a reserved id;
    outside ACC, PAD and the stop ids label no edge except those into ACC."""
    for name, a in _sample_automata():
        w = walker(a)
        acc = w.acc()
        body = w.body_states()
        for src, dst, cls in w.edges:
            if src == acc or dst == acc:
                continue
            forbidden = RESERVED_IDS if src in body else frozenset({PAD, *STOP_IDS})
            hit = w.members_among(cls, forbidden)
            assert not hit, f"{name}: edge {src}->{dst} admits reserved {sorted(hit)}"


# --------------------------------------------------------------------------- adversarial rows


def test_the_companion_answer_is_accepted() -> None:
    """Too-strict guard for the rows below: the answer they corrupt is accepted, compact and pretty."""
    w = walker(compiled_function(LOOKUP_FN))
    assert w.accepts(_frame(LOOKUP_OK))
    assert w.accepts(_frame(LOOKUP_PRETTY))


@pytest.mark.parametrize("reserved", sorted(RESERVED_IDS))
def test_a_reserved_id_inside_a_json_string_body_is_rejected(reserved: int) -> None:
    """``<turn|>`` (106) is an ordinary-looking piece to outlines; inside ``"black"`` it must not match
    (vocab.py: otherwise the canvas is truncated mid-grammar and the next block's state set empties)."""
    body = [*encode('{"user_id": 7890, "special": "bl'), reserved, *encode('ack"}')]
    assert not walker(compiled_function(LOOKUP_FN)).accepts(_frame(body))


@pytest.mark.parametrize("stop", sorted(STOP_IDS))
def test_a_canvas_that_stops_mid_grammar_is_rejected(stop: int) -> None:
    w = walker(compiled_function(LOOKUP_FN))
    cut = [*PHASE0_HEADER, *encode('{"user_id": 78'), stop, *([PAD] * 32)]
    assert not w.accepts(cut)
    closed = [*PHASE0_HEADER, *encode('{"user_id": 78}'), stop, *([PAD] * 32)]
    assert w.accepts(closed), "companion: the same canvas closed before the stop is accepted"


def test_the_nonempty_required_string_option_is_on() -> None:
    """Baseline option ``nonempty_strings``: a required string may not be ``""`` (optional ones may)."""
    fn = {
        "name": "set_label",
        "description": "Set a label.",
        "parameters": {
            "type": "dict",
            "required": ["label"],
            "properties": {"label": {"type": "string", "description": "The label."}},
        },
    }
    w = walker(compiled_function(fn))
    assert w.accepts(_frame('{"label": "x"}'))
    assert not w.accepts(_frame('{"label": ""}'))


def test_enums_are_case_insensitive() -> None:
    fn = {
        "name": "set_mode",
        "description": "Set the mode.",
        "parameters": {
            "type": "dict",
            "required": ["mode"],
            "properties": {
                "mode": {"type": "string", "enum": ["FAST", "SLOW"], "description": "Mode."}
            },
        },
    }
    w = walker(compiled_function(fn))
    for spelling in ("FAST", "fast", "Fast", "slow"):
        assert w.accepts(_frame(f'{{"mode": "{spelling}"}}')), spelling
    assert not w.accepts(_frame('{"mode": "MEDIUM"}'))


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ('{"model": "vikhyatk/moondream2"}', True),
        ('{"model": 1}', True),
        ('{"model": null}', True),
        ('{"model": {"a": 1}}', True),
        ('{"model": [1, "x"]}', True),
        ("1", False),
        ("null", False),
        ('"model"', False),
        ("}", False),
        ("{}", False),
    ],
)
def test_the_expanded_wildcard_is_a_value_inside_the_object(text: str, accepted: bool) -> None:
    """outlines-core 0.2.14 emits ``type: any`` as an unparenthesised 7-way alternation, so the unexpanded
    grammar accepts a bare ``1`` and rejects the object form; ``expand_wildcard`` must repair both."""
    assert walker(compiled_function(ANY_FN)).accepts(_frame(text)) is accepted


def test_the_any_property_of_live_simple_122_78_0() -> None:
    """The named record: its ``model`` property is ``type: any`` and the other two are required strings.
    (The task row reads '{"model": "..."} accepted'; with image_path and question required, that means
    the object carrying a string ``model``.)"""
    w = walker(compiled_bfcl("live_simple_122-78-0"))
    full = (
        '{"image_path": "image.png", "question": "what is shown?", "model": "vikhyatk/moondream2"}'
    )
    assert w.accepts(_frame(full))
    assert not w.accepts(_frame("1"))
    assert not w.accepts(_frame("}"))


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("a", True),
        ("a\nb", True),
        ("a\nb\nb", True),
        ("a\nb\nb\nb", True),
        ("a\nb\nb\nb\nb", False),
        ("a\n", False),
        ("a\nb\n", False),
        ("b", False),
    ],
)
def test_outlines_accept_state_edge_defect_is_worked_around(text: str, accepted: bool) -> None:
    """outlines-core 0.2.14 ``src/index.rs`` drops token edges from an accepting state to a non-accepting
    one; diffgemma-fa's ``lift.anchor_at_eoi`` (``\\z``) works around it. ``a(?:\\nb){0,3}`` then accepts
    ``a\\nb\\nb`` (the edge leaves the accepting state after ``a\\nb``)."""
    a = vendored("pipeline").compile_regex("a(?:\nb){0,3}", name="accept_edge_defect").automaton
    assert walker(a).accepts(_frame(text)) is accepted


def test_declared_property_order_is_the_grammar_order() -> None:
    """The baseline grammar is ordered (diffgemma-fa ``compile_json_schema``: reordered keys are not
    admitted). Pins that the helper and the compiler keep the declared order: ``user_id`` before
    ``special`` is accepted, the reverse is rejected. Guards against a sorted-key memo recurring."""
    w = walker(compiled_function(LOOKUP_FN))
    assert list(LOOKUP_FN["parameters"]["properties"]) == ["user_id", "special"]
    assert w.accepts(_frame('{"user_id": 7890, "special": "black"}'))
    assert not w.accepts(_frame('{"special": "black", "user_id": 7890}'))
