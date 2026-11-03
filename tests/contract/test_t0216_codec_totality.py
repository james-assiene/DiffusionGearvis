"""T-0216: the wire codec stays total on huge int literals and on deep nesting.

``decode`` promises it never raises (docs/03 §3 "Protocol codecs", docs/02 §5 "codec (total
decode)"). Python's ``json.loads`` raises ``ValueError`` on an int literal over the 4300-digit
int-from-string limit and ``RecursionError`` on nesting deeper than the recursion limit, and
neither is a ``JSONDecodeError``. After T-0216 both are ``Malformed`` with a fixed reason that
carries no payload; everything else about the codec is unchanged.

The codec is pure (no service behind it), so these rows have no ``backend`` parametrisation, as
``tests/contract/test_protocol_codec.py`` has none.

Regime: CPython 3.11, default ``sys.get_int_max_str_digits()`` (4300) and recursion limit
(1000); no I/O.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import re
import sys
import tokenize
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from local_astra.contracts import codec
from local_astra.contracts.base import Contract
from local_astra.contracts.codec import MAX_MESSAGE_BYTES, Malformed, Unknown, decode
from local_astra.contracts.protocol import RealtimeEvent

#: Python's default int-from-string limit (docs: ``sys.int_info.default_max_str_digits``).
INT_LIMIT = 4300
#: Well past any plausible recursion limit, well under the 8 MiB size limit.
DEPTH = 100_000


def total_decode(raw: str | bytes) -> object:
    """``decode``, with an escaping exception turned into the assertion it violates (decode is
    total). A regression then fails as an assertion naming the broken promise, not as a bare
    ``ValueError``/``RecursionError`` from inside ``json``."""
    try:
        return decode(raw)
    except Exception as e:
        raise AssertionError(
            f"decode is total (docs/03 §3) but raised {type(e).__name__}: {str(e)[:120]}"
        ) from None


def _digits(n: int, seed: int = 216) -> str:
    """``n`` decimal digits with no leading zero and no long repeat, so a 20-char window of the
    input is distinctive (a reason that quoted the literal would be caught)."""
    rng = random.Random(seed)
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


def _wrap(literal: str, where: str) -> str:
    if where == "top":
        return literal
    if where == "astra":
        return '{"type":"astra.hello","protocol_version":' + literal + "}"
    if where == "realtime":
        return '{"type":"session.update","n":' + literal + "}"
    raise AssertionError(where)


def _deep(kind: str, depth: int = DEPTH) -> str:
    if kind == "array":
        return "[" * depth + "]" * depth
    if kind == "object":
        return '{"a":' * depth + "1" + "}" * depth
    if kind == "inside_payload":
        return '{"type":"x","a":' + "[" * depth + "]" * depth + "}"
    raise AssertionError(kind)


# --- criterion 1 ------------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["top", "astra", "realtime"])
@pytest.mark.parametrize("as_bytes", [False, True], ids=["str", "bytes"])
def test_huge_int_is_malformed(as_bytes: bool, where: str) -> None:
    """Criterion 1: a 4301-digit (and a 5000-digit) literal is Malformed, never a raise."""
    for n in (INT_LIMIT + 1, 5000):
        raw: str | bytes = _wrap(_digits(n), where)
        if as_bytes:
            raw = raw.encode("ascii")
        out = total_decode(raw)
        assert isinstance(out, Malformed), (
            f"criterion 1: a {n}-digit int literal ({where}, {type(raw).__name__}) must decode to "
            f"Malformed; got {out!r:.200}"
        )


def test_4300_digit_int_still_decodes() -> None:
    """Criterion 5, negative and boundary: exactly 4300 digits decodes as today.

    The expected payload is the literal's own value (``int`` of a 4300-digit string is within the
    limit), a guard that is too wide (refusing at 4300) fails here."""
    lit = _digits(INT_LIMIT)
    out = decode(_wrap(lit, "realtime"))
    assert isinstance(out, RealtimeEvent), f"a 4300-digit int is legal; got {out!r:.200}"
    assert out.type == "session.update"
    assert out.payload == {"n": int(lit)}
    # top level: still the existing "not an object" reason, not a new one
    top = decode(lit)
    assert top == Malformed(error="top-level value must be an object"), top


# --- criterion 2 ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["array", "object", "inside_payload"])
def test_deep_nesting_is_malformed(kind: str) -> None:
    """Criterion 2: nesting past the parser's recursion limit is Malformed, str and bytes."""
    raw = _deep(kind)
    assert len(raw.encode()) < MAX_MESSAGE_BYTES  # the size guard must not be what refuses it
    for r in (raw, raw.encode("ascii")):
        out = total_decode(r)
        assert isinstance(out, Malformed), (
            f"criterion 2: {DEPTH}-deep {kind} ({type(r).__name__}) must be Malformed; got "
            f"{out!r:.200}"
        )


# --- criterion 3 ------------------------------------------------------------------------------


def _windows(text: str, size: int = 20) -> set[str]:
    return {text[i : i + size] for i in range(len(text) - size + 1)}


def test_malformed_reason_carries_no_payload() -> None:
    """Criterion 3: the reason is fixed text; no 20-char substring of the input appears in it,
    no digit run or bracket run either, and two different inputs of the same shape get the same
    reason (it can only depend on a digit count or a position)."""
    cases = {
        "huge_int_a": _wrap(_digits(5000, seed=1), "realtime"),
        "huge_int_b": _wrap(_digits(5000, seed=2), "realtime"),
        "deep_array": _deep("array"),
        "deep_object": _deep("object"),
        "deep_payload": _deep("inside_payload"),
    }
    errors: dict[str, str] = {}
    for name, raw in cases.items():
        out = total_decode(raw)
        assert isinstance(out, Malformed), f"{name}: {out!r:.200}"
        err = out.error
        errors[name] = err
        leaked = _windows(err) & _windows(raw)
        assert not leaked, f"criterion 3: {name}'s reason quotes the input: {sorted(leaked)[:3]}"
        assert not re.search(r"\d{50}", err), f"{name}: a digit run in {err!r:.200}"
        assert not re.search(r"[\[\]{}]{2}", err), f"{name}: a bracket run in {err!r:.200}"
        assert len(err) < 300, f"{name}: a reason is a short fixed text, got {len(err)} chars"
    assert errors["huge_int_a"] == errors["huge_int_b"], (
        f"criterion 3: two 5000-digit literals at the same position must give the same fixed "
        f"reason; got {errors['huge_int_a']!r} and {errors['huge_int_b']!r}"
    )


def test_wire_in_row_holds_no_digit_or_bracket_run() -> None:
    """Criterion 3, second sentence: ``hud_server._malformed_row`` of the codec's result holds no
    digit run or bracket run from the input (for astra.settings.set too, which it rewrites)."""
    from local_astra.conductor.hud_server import _malformed_row

    raws = [
        _wrap(_digits(5000), "top"),
        _wrap(_digits(5000), "astra"),
        '{"type":"astra.settings.set","changes":{"a":' + _digits(5000) + "}}",
        _deep("array"),
        '{"type":"astra.settings.set","changes":{"a":' + "[" * DEPTH + "]" * DEPTH + "}}",
    ]
    for raw in raws:
        out = total_decode(raw)
        assert isinstance(out, Malformed), f"{out!r:.200}"
        row = _malformed_row(out)
        assert not re.search(r"\d{20}", row), f"a digit run in the wire.in msg {row!r:.200}"
        assert not re.search(r"[\[\]{}]{2}", row), f"a bracket run in the wire.in msg {row!r:.200}"


# --- criterion 5 ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # hand-checked against json's own messages at main 423362b7: the object is cut after
        # its 11th character, so the parser wants a ',' at position 11.
        ('{"type":"x"', "invalid json: Expecting ',' delimiter at 11"),
        (b'{"type":"x"', "invalid json: Expecting ',' delimiter at 11"),
        ("", "invalid json: Expecting value at 0"),
        ('{"type":"x",}', "invalid json: Expecting property name enclosed in double quotes at 12"),
    ],
)
def test_existing_reasons_unchanged(raw: str | bytes, expected: str) -> None:
    """Criterion 5: a ``JSONDecodeError`` (a ``ValueError`` subclass) keeps its exact text; a fix
    that catches ``ValueError`` first would change it."""
    assert decode(raw) == Malformed(error=expected)


def test_size_check_still_precedes_parsing() -> None:
    """Criterion 5: an oversize message of digits gets the size reason, not a parse reason."""
    big = "1" * (MAX_MESSAGE_BYTES + 1)
    expected = Malformed(error=f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    assert decode(big) == expected
    assert decode(big.encode()) == expected


def test_no_global_limit_change() -> None:
    """Criterion 5: decode does not move ``sys.set_int_max_str_digits`` or the recursion limit,
    at runtime nor (a set-and-restore would hide from the runtime check) in the source."""
    digits, depth = sys.get_int_max_str_digits(), sys.getrecursionlimit()
    for raw in (_wrap(_digits(5000), "realtime"), _deep("array")):
        with contextlib.suppress(ValueError, RecursionError):
            decode(raw)
        assert sys.get_int_max_str_digits() == digits
        assert sys.getrecursionlimit() == depth
    source = Path(codec.__file__).read_text(encoding="utf-8")
    names = {
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.NAME
    }
    forbidden = {"set_int_max_str_digits", "setrecursionlimit"} & names
    assert not forbidden, f"criterion 5: codec.py calls {sorted(forbidden)} (code tokens only)"


# --- adversarial property ---------------------------------------------------------------------

_runs = st.one_of(
    st.integers(1, 6000).map(lambda n: _digits(n, seed=n)),
    st.integers(1, 3000).map(lambda n: "[" * n + "]" * n),
    st.integers(1, 3000).map(lambda n: "[" * n),
    st.integers(1, 3000).map(lambda n: '{"a":' * n),
    st.integers(1, 3000).map(lambda n: "-" + _digits(n, seed=n)),
)


# One explicit example and one reported bug, so a regression reads as a single assertion rather
# than an ExceptionGroup; the strategy reaches past both limits on its own.
@settings(max_examples=40, deadline=None, report_multiple_bugs=False)
@example(run="[" * DEPTH, where="top", as_bytes=True)
@given(
    run=_runs,
    where=st.sampled_from(["top", "astra", "realtime"]),
    as_bytes=st.booleans(),
)
def test_decode_never_raises_on_digit_and_bracket_runs(
    run: str, where: str, as_bytes: bool
) -> None:
    """Criteria 1-2: no digit run or bracket run, at any position, makes decode raise."""
    raw: str | bytes = _wrap(run, where)
    if as_bytes:
        raw = raw.encode("ascii")
    out = total_decode(raw)
    assert isinstance(out, Malformed | Unknown | Contract), f"not a Decoded: {out!r:.200}"
    # Oracle: whatever the stdlib parser refuses (by any exception) must come back Malformed.
    try:
        json.loads(raw)
    except Exception:
        assert isinstance(out, Malformed), f"json.loads refuses it, decode gave {out!r:.200}"
