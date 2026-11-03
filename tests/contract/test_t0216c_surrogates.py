"""T-0216c criteria 1 and 4: ``codec.decode`` replaces every lone surrogate with U+FFFD, and changes nothing else.

A browser's ``JSON.stringify`` emits ``\\ud800`` for broken text (a pasted half emoji). After ``json.loads`` any
surrogate code point left in a Python ``str`` is lone (valid pairs were combined), and it has no UTF-8 encoding,
so everything downstream that encodes it breaks (docs/tasks/T-0216c.md Goal).

Oracles, none of them a copy of the replacement:

- criterion 1: ``decode`` of the frame with ``\\ufffd`` where the lone surrogate escape was (the criterion's own
  wording: "the same message type it would for the text with U+FFFD in its place");
- criterion 4: the codec as it is today, loaded from the literal T-0216c base commit ``T0216C_BASE`` (the
  ``contracts/codec.py`` blob there), on frames with no lone surrogate.
- Hypothesis: lone surrogates injected by the strategy at known places into strings it builds, so the
  expected value is the strategy's own construction with U+FFFD at those places.

Ruling R-1 (T-0216c, from the implementer via the coordinator, 2026-10-31): lone surrogates arrive as JSON
*escapes* in the raw text. A literal U+D800 character inside a Python ``str`` handed to ``decode`` cannot come off
the wire (text frames are valid UTF-8) and keeps today's outcome
(``tests/contract/test_protocol_codec.py::test_a_short_unencodable_string_keeps_todays_outcome``); no row here
injects one.

Regime: pure codec calls, no I/O beyond ``git show`` of one blob; Hypothesis default profile.
"""

from __future__ import annotations

import functools
import itertools
import json
import subprocess
import sys
import threading
import types
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from local_astra.contracts import codec
from local_astra.contracts.codec import Malformed, Unknown, decode, encode
from local_astra.contracts.protocol import (
    ASTRA_MESSAGE_TYPES,
    AstraText,
    AstraTranscript,
    AstraUiAction,
    RealtimeEvent,
)
from tests import _git_range as gr
from tests.unit._examples import example_of
from tests.unit.test_hud_constants_parity import corpus_cases, raw_of

pytestmark = [pytest.mark.contract]

#: T-0216c's literal base (main at the plan commit, which includes T-0216): ``contracts/codec.py`` there is the
#: codec "as today" for criterion 4's negatives.
T0216C_BASE = "7489b9eda75f382f88ce62e2a0b56ae4719d3a29"
CODEC_PATH = "local_astra/contracts/codec.py"

LONE = "\\ud800"  # the JSON escape, six ASCII characters, as the page puts it on the wire
FFFD = "\\ufffd"


def _full_sha(short: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(gr.REPO), "rev-parse", "--verify", "--quiet", short + "^{commit}"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


@functools.lru_cache(maxsize=1)
def old_codec() -> types.ModuleType:
    """``contracts/codec.py`` as it is at ``T0216C_BASE``, executed as its own module (same protocol classes)."""
    gr.require_commits(T0216C_BASE)
    src = subprocess.run(
        ["git", "-C", str(gr.REPO), "show", f"{T0216C_BASE}:{CODEC_PATH}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    module = types.ModuleType("t0216c_base_codec")
    module.__file__ = f"<{T0216C_BASE[:8]}:{CODEC_PATH}>"
    sys.modules.setdefault(module.__name__, module)
    exec(compile(src, module.__file__, "exec"), module.__dict__)
    return module


def _same(new: Any, old: Any) -> bool:
    """Equality across the two codec modules: ``Malformed``/``Unknown`` are each module's own dataclasses, so they
    compare by class name and fields; every protocol model is the one shared class and compares as itself."""
    if type(new).__name__ in ("Malformed", "Unknown"):
        return type(old).__name__ == type(new).__name__ and vars(old) == vars(new)
    return bool(type(new) is type(old) and new == old)


def _surrogates(s: str) -> list[str]:
    return [c for c in s if 0xD800 <= ord(c) <= 0xDFFF]


def _all_strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in _all_strings(k) + _all_strings(v)]
    if isinstance(obj, list | tuple):
        return [s for v in obj for s in _all_strings(v)]
    return []


def _strings_of(decoded: Any) -> list[str]:
    if isinstance(decoded, RealtimeEvent):
        return [decoded.type, *_all_strings(decoded.payload)]
    if isinstance(decoded, Unknown | Malformed):
        return [s for s in vars(decoded).values() if isinstance(s, str)]
    return _all_strings(decoded.model_dump())


def _pair(template: str, lone: str = LONE) -> tuple[str, str]:
    """(the frame with the lone surrogate escape at each ``@S@``, the same frame with ``\\ufffd`` there)."""
    return template.replace("@S@", lone), template.replace("@S@", FFFD)


# --- criterion 1: the replacement, at every position, depth, field kind and message family -------------------

#: (id, template with @S@ where a lone surrogate escape goes, expected decoded kind)
CASES: list[tuple[str, str, type]] = [
    ("text_middle", '{"type":"astra.text","text":"a@S@ hi"}', AstraText),
    ("text_start", '{"type":"astra.text","text":"@S@a hi"}', AstraText),
    ("text_end", '{"type":"astra.text","text":"a hi@S@"}', AstraText),
    ("text_only", '{"type":"astra.text","text":"@S@"}', AstraText),
    ("text_two_lone_in_a_row", '{"type":"astra.text","text":"a@S@@S@b"}', AstraText),
    (
        "transcript_text",
        '{"type":"astra.transcript","turn_id":"t@S@1","text":"x@S@y"}',
        AstraTranscript,
    ),
    (
        "ui_action_context_key_and_value",
        '{"type":"astra.ui.action","surface_id":"s","name":"n@S@","context":{"k@S@":"v@S@","n":1}}',
        AstraUiAction,
    ),
    (
        "ui_action_deep_data_model",
        '{"type":"astra.ui.action","surface_id":"s","name":"n","context":{},'
        '"data_model":{"a":[{"b":{"c":[[["@S@"]]],"@S@d":{"e@S@":null}}}]}}',
        AstraUiAction,
    ),
    ("realtime_value", '{"type":"session.update","session":{"voice":"x@S@"}}', RealtimeEvent),
    ("realtime_top_level_key", '{"type":"session.update","@S@":1,"event_id":"e"}', RealtimeEvent),
    (
        "realtime_deep_key_and_list",
        '{"type":"conversation.item.create","item":{"a":{"b":{"c":{"d@S@":[1,["x","@S@"],{"@S@":"@S@"}]}}}}}',
        RealtimeEvent,
    ),
    (
        "realtime_type_itself",
        '{"type":"session.up@S@date","x":1}',
        Malformed,
    ),  # the type pattern refuses U+FFFD too
    ("unknown_astra_type", '{"type":"astra.fu@S@ture","x":"@S@"}', Unknown),
]

#: every surrogate block boundary and hex-case spelling of the escape (the criterion-4 prefilter is
#: case-insensitive over ``\\ud8``-``\\udf``)
LONE_SPELLINGS = [
    "\\ud800",
    "\\uD800",
    "\\udbff",
    "\\uDBFF",
    "\\udc00",
    "\\uDc00",
    "\\udfff",
    "\\uDFFF",
    "\\ud9ab",
]


@pytest.mark.parametrize("as_bytes", [False, True], ids=["str", "bytes"])
@pytest.mark.parametrize(("template", "kind"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_a_lone_surrogate_decodes_as_u_fffd_in_its_place(
    template: str, kind: type, as_bytes: bool
) -> None:
    """Criterion 1: values and keys, any position or depth, astra and Realtime types, str and bytes frames."""
    bad, good = _pair(template)
    want = decode(good)
    assert isinstance(want, kind), (
        f"premise: the U+FFFD frame decodes as {kind.__name__}; got {want!r}"
    )
    got = decode(bad.encode() if as_bytes else bad)
    assert got == want, (
        f"criterion 1: decode({bad!r}) returns what decode of the U+FFFD frame returns ({want!r}); got {got!r}"
    )
    lone = [s for s in _strings_of(got) if _surrogates(s)]
    assert lone == [], f"criterion 1: no string of the decoded value holds a surrogate; {lone!r}"
    if not isinstance(got, Unknown | Malformed):
        encode(got).encode("utf-8")  # the echo path (HudServer.publish) must not raise


@pytest.mark.parametrize("lone", LONE_SPELLINGS)
def test_every_spelling_of_a_lone_surrogate_escape_is_replaced(lone: str) -> None:
    """Criterion 1 over the whole U+D800-U+DFFF range and both hex cases (high and low halves, lone)."""
    bad, good = _pair('{"type":"astra.text","text":"a@S@ hi"}', lone)
    got = decode(bad)
    assert got == decode(good) == AstraText(text="a� hi"), (
        f"criterion 1: {lone} alone decodes as U+FFFD; got {got!r}"
    )


@pytest.mark.parametrize(
    ("escapes", "text"),
    [
        ("\\udc00\\ud800", "��"),  # low then high: not a pair, two lone halves
        ("\\ud800\\ud800\\udc00", "�\U00010000"),  # a lone high, then a valid pair
        ("\\ud800\\u0041", "�A"),  # a high followed by a non-surrogate escape
        ("\\ud83d\\ude00\\udc00", "\U0001f600�"),  # a valid pair, then a lone low
        ("\\ud800\\ud83d\\ude00", "�\U0001f600"),  # a lone high before a valid pair
    ],
    ids=[
        "low_high",
        "lone_high_then_pair",
        "high_then_ascii_escape",
        "pair_then_lone_low",
        "lone_high_then_emoji",
    ],
)
def test_lone_halves_next_to_valid_pairs(escapes: str, text: str) -> None:
    """Criterion 1: a lone half is replaced and an adjacent valid pair is kept (expected values by hand)."""
    got = decode('{"type":"astra.text","text":"' + escapes + '"}')
    assert got == AstraText(text=text), f"criterion 1: {escapes} decodes as {text!r}; got {got!r}"


@pytest.mark.parametrize(
    ("escapes", "text"),
    [
        ("\\ud83d\\ude00", "\U0001f600"),
        ("\\uD83D\\uDE00", "\U0001f600"),
        ("\\ud800\\udc00", "\U00010000"),
        ("\\udbff\\udfff", "\U0010ffff"),
        ("a\\ud83d\\ude00b\\ud83c\\udf89", "a\U0001f600b\U0001f389"),
    ],
    ids=["emoji", "emoji_upper", "first_astral", "last_astral", "two_emoji"],
)
def test_valid_surrogate_pairs_are_untouched(escapes: str, text: str) -> None:
    """Criterion 1, the guard's too-wide side: a valid pair is one astral character, never U+FFFD U+FFFD.

    Green today and must stay green: a replacement applied to the raw escapes before ``json.loads`` (or to every
    surrogate escape) would turn each of these into two replacement characters."""
    for raw in (
        '{"type":"astra.text","text":"' + escapes + '"}',
        '{"type":"x.y","k' + escapes + '":"' + escapes + '"}',
    ):
        got = decode(raw)
        assert "�" not in "".join(_strings_of(got)), (
            f"criterion 1: {raw!r} kept its pairs; got {got!r}"
        )
    assert decode('{"type":"astra.text","text":"' + escapes + '"}') == AstraText(text=text)
    ev = decode('{"type":"x.y","k' + escapes + '":"' + escapes + '"}')
    assert isinstance(ev, RealtimeEvent) and ev.payload == {"k" + text: text}


# --- criterion 1, "any nesting depth": a leaf ~990 deep (T-0216c review) -----------------------------------

#: The deepest nesting asked for (T-0216b's rows stay <= 990 too): Python 3.11's recursion limit is 1000 and the
#: C JSON scanner spends one level of it per container. A frame this deep is ~6 KB, far under ``MAX_MESSAGE_BYTES``.
DEEP_CAP = 990
#: The row's premise: in a fresh thread the parser accepts at least this depth (else the row proves little).
DEEP_FLOOR = 980


def _deep_frame(family: str, depth: int, leaf: str) -> str:
    """A frame whose single string leaf sits under ``depth`` nested containers (alternating dict and list)."""
    opens = "".join('{"k":' if i % 2 == 0 else "[" for i in range(depth))
    closes = "".join("}" if i % 2 == 0 else "]" for i in reversed(range(depth)))
    body = opens + '"' + leaf + '"' + closes
    if family == "realtime":
        return '{"type":"conversation.item.create","item":' + body + "}"
    return (
        '{"type":"astra.ui.action","surface_id":"s","name":"n","context":{},"data_model":'
        + body
        + "}"
    )


def _deep_leaf(root: Any, depth: int) -> Any:
    """The leaf under ``depth`` alternating containers, walked iteratively (no recursion in the oracle)."""
    node = root
    for i in range(depth):
        node = node["k"] if i % 2 == 0 else node[0]
    return node


def _in_fresh_thread(fn: Any) -> Any:
    """Run ``fn`` on a new thread, so the call depth is a few frames whatever pytest's (or xdist's) stack is."""
    out: dict[str, Any] = {}

    def run() -> None:
        try:
            out["value"] = fn()
        except BaseException as exc:  # handed back to the test thread
            out["exc"] = exc

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "exc" in out:
        raise out["exc"]
    return out["value"]


@pytest.mark.parametrize("as_bytes", [False, True], ids=["str", "bytes"])
@pytest.mark.parametrize("family", ["realtime", "ui_action_data_model"])
def test_a_lone_surrogate_at_the_bottom_of_a_deep_frame_is_replaced(
    family: str, as_bytes: bool
) -> None:
    """Criterion 1, "at any position or nesting depth" (T-0216c review): a lone-surrogate escape at a leaf as deep
    as the parser accepts (>= 980, <= 990 containers; a fresh thread) decodes, without raising and not
    Malformed, to the family's message with U+FFFD at that leaf.

    Depth: the deepest ``d <= DEEP_CAP`` whose U+FFFD frame (no surrogate escape, so no walk) still decodes as
    the family's type on the same thread; that is the premise, so the row tests the walk, not the parser."""
    kind = RealtimeEvent if family == "realtime" else AstraUiAction

    def body() -> tuple[int, Any]:
        depth = DEEP_CAP
        while depth >= DEEP_FLOOR and not isinstance(
            decode(_deep_frame(family, depth, FFFD)), kind
        ):
            depth -= 1
        if depth < DEEP_FLOOR:
            return depth, None
        bad = _deep_frame(family, depth, LONE)
        try:
            return depth, decode(bad.encode() if as_bytes else bad)
        except RecursionError as exc:
            return depth, exc

    depth, got = _in_fresh_thread(body)
    assert depth >= DEEP_FLOOR, (
        f"premise: in a fresh thread the parser accepts a {DEEP_FLOOR}-deep U+FFFD frame; it did not"
    )
    assert not isinstance(got, RecursionError), (
        f"criterion 1: decode of a {depth}-deep lone-surrogate frame raised RecursionError "
        f"(the parser accepted this depth; a recursive walk of the value?)"
    )
    assert isinstance(got, kind), (
        f"criterion 1: the {depth}-deep lone-surrogate frame decodes as {kind.__name__}; "
        f"got {str(got)[:200]}"
    )
    root = got.payload["item"] if isinstance(got, RealtimeEvent) else got.data_model
    leaf = _deep_leaf(root, depth)
    assert leaf == "\ufffd", f"criterion 1: the leaf {depth} deep reads U+FFFD; got {leaf!r}"


# --- criterion 1, Hypothesis: random strings with injected lone surrogates ----------------------------------

_HIGH = st.integers(0xD800, 0xDBFF).map(chr)
_LOW = st.integers(0xDC00, 0xDFFF).map(chr)
_LONE_CHAR = st.one_of(_HIGH, _LOW)


@st.composite
def injected(draw: st.DrawFn) -> tuple[str, str]:
    """(a string with lone surrogates injected, the same string with U+FFFD at exactly those places).

    The base text never holds a surrogate (Hypothesis' ``text`` excludes category Cs). An injected high half
    immediately followed by an injected low half would be a *valid pair* once JSON-escaped, so that adjacency
    is rejected; everything else stays lone after ``json.loads``."""
    parts = draw(
        st.lists(
            st.tuples(st.text(max_size=6), st.lists(_LONE_CHAR, max_size=3)), min_size=1, max_size=4
        )
    )
    with_lone = "".join(t + "".join(s) for t, s in parts)
    expected = "".join(t + "�" * len(s) for t, s in parts)
    for a, b in itertools.pairwise(with_lone):
        assume(not (0xD800 <= ord(a) <= 0xDBFF and 0xDC00 <= ord(b) <= 0xDFFF))
    return with_lone, expected


_LEAF = st.one_of(st.none(), st.booleans(), st.integers(-10, 10), injected())


def _tree() -> st.SearchStrategy[Any]:
    return st.recursive(
        _LEAF,
        lambda kids: st.one_of(
            st.lists(kids, max_size=3),
            st.lists(st.tuples(injected(), kids), max_size=3),  # a dict as (key pair, value) items
        ),
        max_leaves=12,
    )


def _split(node: Any) -> tuple[Any, Any]:
    """(the value carrying lone surrogates, the expected value): ``injected`` pairs are split, dict items built."""
    if isinstance(node, tuple) and len(node) == 2 and all(isinstance(x, str) for x in node):
        return node
    if (
        isinstance(node, list)
        and node
        and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], tuple) for x in node)
    ):
        bad: dict[str, Any] = {}
        good: dict[str, Any] = {}
        for (kb, kg), v in node:
            vb, vg = _split(v)
            if kb in bad or kg in good:
                continue  # two keys that collide on either side: keep the first on both
            bad[kb], good[kg] = vb, vg
        return bad, good
    if isinstance(node, list):
        pairs = [_split(x) for x in node]
        return [b for b, _ in pairs], [g for _, g in pairs]
    return node, node


@given(tree=_tree(), key=injected())
def test_hypothesis_injected_lone_surrogates_become_u_fffd(tree: Any, key: tuple[str, str]) -> None:
    """Criterion 1: for any value with lone surrogates injected (as JSON escapes: ``ensure_ascii``), decode never
    raises, the payload equals the construction with U+FFFD in their places, and the result encodes to UTF-8."""
    bad_value, good_value = _split(tree)
    assume(key[0] != "type")
    raw = json.dumps({"type": "rt.event", key[0]: bad_value}, ensure_ascii=True)
    assert not _surrogates(raw), (
        "premise (R-1): every surrogate is a JSON escape in the raw text, never a character"
    )
    got = decode(raw)
    assert isinstance(got, RealtimeEvent), f"criterion 1: a Realtime frame stays one; got {got!r}"
    assert got.payload == {key[1]: good_value}, (
        f"criterion 1: every lone surrogate, keys included, becomes U+FFFD; got {got.payload!r}"
    )
    encode(got).encode("utf-8")


@given(text=injected())
def test_hypothesis_typed_text_with_lone_surrogates(text: tuple[str, str]) -> None:
    """Criterion 1 on ``astra.text`` (the typed-text path of the bug): decode equals decode of the U+FFFD text."""
    bad, good = text
    assume(good.strip() and len(good.strip()) <= 2000)
    got = decode(json.dumps({"type": "astra.text", "text": bad}, ensure_ascii=True))
    assert got == AstraText(text=good), f"criterion 1: {bad!r} decodes as {good!r}; got {got!r}"
    encode(got).encode("utf-8")


# --- criterion 4: no lone surrogate -> exactly today's decode; encode unchanged ------------------------------


@pytest.mark.parametrize("case", corpus_cases(), ids=lambda c: str(c.get("name")))
def test_the_decode_corpus_decodes_as_today(case: dict[str, Any]) -> None:
    """Criterion 4: every row of the shared corpus (``tests/fixtures/hud/decode_corpus.jsonl``) decodes to what
    the ``T0216C_BASE`` codec returns, as str and as bytes."""
    raw = raw_of(case)
    assume_no_lone = not any(0xD800 <= ord(c) <= 0xDFFF for c in raw)
    if not assume_no_lone:
        pytest.skip("R-1: a literal surrogate character in a str is out of T-0216c's scope")
    old = old_codec()
    assert _same(decode(raw), old.decode(raw)), f"criterion 4: {case.get('name')} decodes as today"
    assert _same(decode(raw.encode()), old.decode(raw.encode())), (
        f"criterion 4: {case.get('name')} (bytes) as today"
    )


@pytest.mark.parametrize("model", ASTRA_MESSAGE_TYPES, ids=lambda m: m.__name__)
def test_encode_and_round_trip_are_unchanged(model: type) -> None:
    """Criterion 4: encode is unchanged and every example message decodes exactly as today."""
    old = old_codec()
    ex = example_of(model)
    wire = encode(ex)
    assert wire == old.encode(ex), (
        f"criterion 4: encode of {model.__name__} is byte-for-byte today's"
    )
    assert _same(decode(wire), old.decode(wire)) and decode(wire) == ex


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"astra.text","text":"\\\\ud800 is how you write it"}',  # an escaped backslash: literal text
        '{"type":"astra.text","text":"a\\\\uD83D\\\\uDE00"}',
        '{"type":"x.y","k":"\\\\udfff","n":[1,2,{"\\\\ud8":"\\\\udc"}]}',
        '{"type":"astra.text","text":"\\\\ud800',  # truncated: Malformed as today
        '{"type":"astra.text","text":"\\ud800',  # truncated with a real escape: still Malformed, same error
        '[{"type":"astra.text","text":"\\ud800"}]',  # not an object: Malformed as today
        '{"type":"astra.text","text":"\\ufffd"}',
        '{"type":"astra.text","text":"\\u00e9\\u20ac\\ud83d\\ude00"}',
    ],
    ids=[
        "escaped_backslash",
        "escaped_backslash_pair",
        "escaped_in_keys",
        "truncated_plain",
        "truncated_escape",
        "array_top",
        "fffd_itself",
        "bmp_and_pair_escapes",
    ],
)
def test_frames_without_a_lone_surrogate_decode_as_today(raw: str) -> None:
    """Criterion 4, adversarial: text that *looks* like a surrogate escape (``\\\\ud800`` after an escaped
    backslash) is literal text and survives; frames that are Malformed today stay Malformed with the same error.
    A replacement done on the raw text instead of the decoded strings would corrupt the first three."""
    old = old_codec()
    assert _same(decode(raw), old.decode(raw)), (
        f"criterion 4: {raw!r} decodes as today: {old.decode(raw)!r}"
    )


_JSON = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), st.text()),
    lambda kids: st.one_of(
        st.lists(kids, max_size=4), st.dictionaries(st.text(max_size=5), kids, max_size=4)
    ),
    max_leaves=16,
)


@given(
    payload=st.dictionaries(st.text(max_size=5).filter(lambda k: k != "type"), _JSON, max_size=4),
    kind=st.sampled_from(["rt.event", "astra.text", "astra.future", "session.update"]),
    ascii_only=st.booleans(),
)
def test_hypothesis_frames_without_lone_surrogates_decode_as_today(
    payload: dict[str, Any], kind: str, ascii_only: bool
) -> None:
    """Criterion 4: random surrogate-free frames (astral characters included, escaped as valid pairs when
    ``ensure_ascii``) decode to exactly what the base codec returns."""
    raw = json.dumps({"type": kind, **payload}, ensure_ascii=ascii_only)
    assert _same(decode(raw), old_codec().decode(raw))


class _TouchSpy(dict):  # type: ignore[type-arg]
    """A nested mapping that records whether anything walked into it."""

    touched = False

    def _note(self) -> None:
        type(self).touched = True

    def __iter__(self):  # type: ignore[no-untyped-def]
        self._note()
        return super().__iter__()

    def items(self):  # type: ignore[no-untyped-def]
        self._note()
        return super().items()

    def keys(self):  # type: ignore[no-untyped-def]
        self._note()
        return super().keys()

    def values(self):  # type: ignore[no-untyped-def]
        self._note()
        return super().values()


def test_a_frame_without_a_surrogate_escape_is_not_walked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 4: "no second walk when the raw text has no surrogate escape". ``json.loads`` is replaced by a
    stand-in that returns a Realtime object whose nested mapping records being iterated; the raw text has no
    ``\\ud8``-``\\udf`` escape, so the codec must not walk into it (nor re-parse with a hook)."""
    _TouchSpy.touched = False
    calls: list[dict[str, Any]] = []
    nested = _TouchSpy({"voice": "x"})

    def loads(s: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return {"type": "session.update", "session": nested}

    monkeypatch.setattr(codec.json, "loads", loads)
    got = decode('{"type":"session.update","session":{"voice":"x"}}')
    monkeypatch.undo()
    assert isinstance(got, RealtimeEvent), f"premise: the stand-in's object is decoded; got {got!r}"
    assert calls and all(not kw for kw in calls), (
        f"criterion 4: plain json.loads, no hook; calls {calls}"
    )
    assert len(calls) == 1, f"criterion 4: one parse per frame; {len(calls)}"
    assert _TouchSpy.touched is False, (
        "criterion 4: a frame with no surrogate escape is not walked a second time"
    )


def test_t0216c_base_is_the_literal_commit() -> None:
    """Provenance of ``T0216C_BASE``: the literal resolves to itself (a commit object, not a moving ref)."""
    gr.require_commits(T0216C_BASE)
    assert _full_sha(T0216C_BASE) == T0216C_BASE
