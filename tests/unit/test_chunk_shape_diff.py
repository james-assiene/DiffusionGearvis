"""T-0095 criterion 5: the mismatch message of T-0088 gpu row 2 names the first differing chunk index and every
differing key path in that chunk with both values in full. Expected strings are written by hand."""

from __future__ import annotations

from tests.chunk_shape_diff import first_mismatch

LONG = "x" * 2000 + "END"


def _chunk(**choice: object) -> dict[str, object]:
    return {"object": "chat.completion.chunk", "choices": [{"index": 0, **choice}]}


def test_missing_extra_and_differing_value_are_each_named_in_full() -> None:
    role = _chunk(delta={"role": "assistant", "content": ""})
    built = [
        role,
        _chunk(delta={"content": "one two"}, token_ids=[7, 8], finish_reason=None),
    ]
    live = [
        role,
        _chunk(
            delta={"content": "one " + LONG}, token_ids=[7, 8], stop_reason=None
        ),  # finish_reason missing, stop_reason extra
    ]
    msg = first_mismatch(built, live)
    assert msg is not None
    assert "first differing chunk index 1" in msg
    lines = msg.splitlines()
    assert "  choices[0].finish_reason: None vs <absent>" in lines  # missing on the live side
    assert "  choices[0].stop_reason: <absent> vs None" in lines  # extra on the live side
    assert (
        f"  choices[0].delta.content: 'one two' vs 'one {LONG}'" in lines
    )  # in full, not truncated
    assert not any("token_ids" in ln for ln in lines), "an equal key is not named"
    assert not any("index 0" in ln for ln in lines), "chunk 0 is equal"


def test_equal_lists_give_no_message_and_types_and_order_count() -> None:
    a = [_chunk(delta={"content": "a"}, token_ids=[1])]
    assert first_mismatch(a, [_chunk(delta={"content": "a"}, token_ids=[1])]) is None
    # JSON equality, as the row's json.dumps comparison sees it: 1 vs 1.0 differ, and so does key order
    num = first_mismatch([{"n": 1}], [{"n": 1.0}])
    assert num is not None and "  n: 1 vs 1.0" in num.splitlines()
    order = first_mismatch([{"a": 1, "b": 2}], [{"b": 2, "a": 1}])
    assert order is not None and "key order" in order and "['a', 'b'] vs ['b', 'a']" in order


def test_a_missing_chunk_is_named_by_index() -> None:
    a = [_chunk(delta={"content": "a"}), _chunk(delta={"content": "b"})]
    msg = first_mismatch(a, a[:1])
    assert msg is not None
    assert "first differing chunk index 1" in msg and "vs <absent chunk>" in msg
    assert "built 2 vs live 1" in msg
