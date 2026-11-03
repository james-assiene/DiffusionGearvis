"""The page's constants must equal the Python policy's (T-0008 criterion 5, and criterion 2).

"The bucket/quality/fps constants used by the page are **compared against the Python values** by
a test, so a divergence fails rather than silently changing what the model sees"
(``.claude/rules/perception-memory.md``: "changing a bucket requires re-running B2").

This module parses the constants out of the page's source instead of importing it, because
``hud/`` is buildless ES modules with no node dependency in the fast suite. It therefore pins the
*names* the page must use:

    hud/src/frames.js:    SCREEN_MAX = { name: "screen_max", max_long_side: N, max_short_side: N }
                          CAMERA_BUCKET = { name: "camera", max_long_side: N, max_short_side: N }
                          JPEG_QUALITY = N       MAX_FPS = N       RING_BUFFER_S = N
    hud/src/protocol.js:  MAX_MESSAGE_BYTES = N  PROTOCOL_VERSION = N

It also checks the shared decode corpus (``tests/fixtures/hud/decode_corpus.jsonl``, read by
``tests/hud/protocol.test.mjs``) against the **real** ``codec.decode``, so the JS mirror is
compared to the contract and never to itself.

Rows whose payload is megabytes long (T-0027: the multibyte oversize cases and the two rows on
either side of exactly ``MAX_MESSAGE_BYTES``) carry a ``build`` spec instead of ``raw`` and are
constructed at read time by ``build_raw`` below, which is the Python half of the builder contract
in ``tests/fixtures/hud/MANIFEST.md``; ``buildRaw`` in ``tests/hud/protocol.test.mjs`` is the
other half. A row with a ``"pending"`` key runs as strict xfail until its task lands.
"""

from __future__ import annotations

import json
import re
import typing
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.codec import MAX_MESSAGE_BYTES, Malformed, Unknown, decode
from local_astra.contracts.perception import AssistFlags, ResolutionBucket
from local_astra.contracts.protocol import (
    CLIENT_TO_SERVER,
    PROTOCOL_VERSION,
    SERVER_TO_CLIENT,
    RealtimeEvent,
)
from local_astra.perception import assists, policy

REPO = Path(__file__).resolve().parents[2]
FRAMES_JS = REPO / "hud" / "src" / "frames.js"
PROTOCOL_JS = REPO / "hud" / "src" / "protocol.js"
CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "hud" / "decode_corpus.jsonl"


def read_page_module(path: Path, criterion: str) -> str:
    assert path.exists(), (
        f"T-0008 {criterion} adds {path.relative_to(REPO)} — the page mirrors the Python "
        "constants and this test is what stops the two drifting apart"
    )
    return path.read_text()


def number(src: str, name: str, path: Path) -> float:
    """Read ``name = <number>`` out of a JS module (``const``/``let``/``export const``)."""
    m = re.search(rf"\b{name}\s*[:=]\s*(-?\d+(?:\.\d+)?)", src)
    assert m is not None, (
        f"T-0008 criterion 5: {path.name} must define {name} as a plain number so this parity "
        f"test can compare it with the Python value; not found in {path.relative_to(REPO)}"
    )
    return float(m.group(1))


def bucket(src: str, name: str, path: Path) -> dict[str, Any]:
    """Read ``name = {{ name: "x", max_long_side: N, max_short_side: N }}`` out of a JS module."""
    m = re.search(rf"\b{name}\s*=\s*\{{(.*?)\}}", src, re.DOTALL)
    assert m is not None, (
        f"T-0008 criterion 5: {path.name} must define {name} as an object literal with name, "
        "max_long_side and max_short_side (the Python ResolutionBucket fields the page needs)"
    )
    body = m.group(1)
    label = re.search(r"""\bname\s*:\s*["']([^"']+)["']""", body)
    assert label is not None, f"{name} in {path.name} has no name: field"
    return {
        "name": label.group(1),
        "max_long_side": int(number(body, "max_long_side", path)),
        "max_short_side": int(number(body, "max_short_side", path)),
    }


def assert_bucket_matches(got: dict[str, Any], want: ResolutionBucket, which: str) -> None:
    assert got["name"] == want.name, f"criterion 5: {which} name drifted: {got} vs {want.name}"
    assert got["max_long_side"] == want.max_long_side, (
        f"criterion 5: {which} max_long_side is {got['max_long_side']} in hud/src/frames.js and "
        f"{want.max_long_side} in perception/policy.py — one of the two changed without the other "
        "(rules/perception-memory.md: changing a bucket requires re-running B2)"
    )
    assert got["max_short_side"] == want.max_short_side, (
        f"criterion 5: {which} max_short_side is {got['max_short_side']} in hud/src/frames.js and "
        f"{want.max_short_side} in perception/policy.py"
    )


def test_the_screen_bucket_on_the_page_equals_policy_screen_buckets_0():
    """Criterion 5: screen frames use ``policy.SCREEN_BUCKETS[0]`` (pixel-first, docs/12 §3)."""
    src = read_page_module(FRAMES_JS, "criterion 5")
    assert_bucket_matches(bucket(src, "SCREEN_MAX", FRAMES_JS), policy.SCREEN_BUCKETS[0], "screen")


def test_the_camera_bucket_on_the_page_equals_policy_camera_bucket():
    """Criterion 5."""
    src = read_page_module(FRAMES_JS, "criterion 5")
    assert_bucket_matches(bucket(src, "CAMERA_BUCKET", FRAMES_JS), policy.CAMERA_BUCKET, "camera")


@pytest.mark.parametrize(
    ("name", "expected"),
    [("JPEG_QUALITY", policy.JPEG_QUALITY), ("MAX_FPS", policy.MAX_FPS)],
)
def test_the_quality_and_rate_constants_on_the_page_equal_the_policy(name, expected):
    """Criterion 5: quality 85 and 1 fps are policy, not page opinions."""
    src = read_page_module(FRAMES_JS, "criterion 5")
    assert number(src, name, FRAMES_JS) == float(expected), (
        f"criterion 5: {name} is {number(src, name, FRAMES_JS)} in hud/src/frames.js and "
        f"{expected} in local_astra/perception/policy.py — parity is the whole point of this test"
    )


def test_the_page_mirrors_the_codec_message_limit_and_the_protocol_version():
    """Criterion 2 and 1: the page mirrors ``codec.MAX_MESSAGE_BYTES`` and ``PROTOCOL_VERSION``."""
    src = read_page_module(PROTOCOL_JS, "criteria 1-2")
    assert number(src, "MAX_MESSAGE_BYTES", PROTOCOL_JS) == float(MAX_MESSAGE_BYTES), (
        "criterion 2: a message over codec.MAX_MESSAGE_BYTES is Malformed on both sides"
    )
    assert number(src, "PROTOCOL_VERSION", PROTOCOL_JS) == float(PROTOCOL_VERSION), (
        "criterion 1: astra.hello carries protocol_version == PROTOCOL_VERSION"
    )


def string_set(src: str, name: str, path: Path) -> set[str]:
    """Read the string literals out of ``name = [...]`` / ``new Set([...])`` in a JS module."""
    m = re.search(rf"\b{name}\s*=\s*(?:new\s+Set\s*\()?\s*\[(.*?)\]", src, re.DOTALL)
    assert m is not None, (
        f"T-0008 criteria 1-2: {path.name} must define {name} as an array (or Set) of type "
        "strings, so this test can compare it with local_astra/contracts/protocol.py"
    )
    return set(re.findall(r"""["']([^"']+)["']""", m.group(1)))


def test_the_pages_server_to_client_set_equals_the_contract():
    """Criterion 2: "known astra.* in protocol.SERVER_TO_CLIENT -> typed dispatch"."""
    src = read_page_module(PROTOCOL_JS, "criterion 2")
    assert string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS) == set(SERVER_TO_CLIENT), (
        "criterion 2: the page's dispatch table and contracts/protocol.py SERVER_TO_CLIENT must "
        "be the same set; adding a message type means touching both (CLAUDE.md: changing a wire "
        "format means changing the fake in the same PR)"
    )


def test_the_played_message_is_in_both_server_to_client_lists():
    """T-0030 criteria 1 and 21: ``astra.spoken.played`` is in the contract and on the page.

    The test above compares the two lists as *sets*, which is what makes them one list; this row
    names the type, so that "the page and the contract agree" cannot be satisfied by both of them
    forgetting it. It is also the marker of the task's ordering: group A lands the Python half and
    group C the page's, and between those two merges the set-equality test above is red **on
    purpose** (docs/tasks/T-0030.md implementation notes, "four existing green tests go red on
    purpose"). Any other red test in this module is a behaviour bug.
    """
    src = read_page_module(PROTOCOL_JS, "T-0030 criterion 21")
    assert "astra.spoken.played" in set(SERVER_TO_CLIENT), (
        "T-0030 criterion 1: contracts/protocol.py SERVER_TO_CLIENT gains astra.spoken.played"
    )
    assert "astra.spoken.played" in string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS), (
        "T-0030 criterion 21: hud/src/protocol.js mirrors it, or the page drops the message that "
        "tells it which words were never heard"
    )
    assert "astra.spoken.played" not in set(CLIENT_TO_SERVER), (
        "criterion 1: server->client only; a page that sends it is refused as wrong_direction"
    )


def test_the_page_emits_only_hello_and_frame_and_they_are_client_to_server_types():
    """Criterion 1: the whitelist, and that it is a subset of what the contract allows a client."""
    src = read_page_module(PROTOCOL_JS, "criterion 1")
    client_types = string_set(src, "CLIENT_TYPES", PROTOCOL_JS)
    assert client_types == {"astra.hello", "astra.frame"}, (
        "criterion 1 negative: astra.frame.text and astra.assist.toggle belong to T-0012 and "
        f"astra.ui.action to T-0010; this task's code emits neither, got {sorted(client_types)}"
    )
    assert client_types <= set(CLIENT_TO_SERVER)


# --- the decode corpus the JS mirror is tested against --------------------------------------------


#: The envelope the boundary rows are built from (49 + 2 bytes, hand-counted).
PREFIX = '{"type":"astra.transcript","turn_id":"t","text":"'
SUFFIX = '"}'
BOUNDARY_SPEC: dict[str, Any] = {
    "prefix": PREFIX,
    "char": "x",
    "suffix": SUFFIX,
    "fill_to_bytes": MAX_MESSAGE_BYTES,
}


def corpus_cases() -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in CORPUS.read_text().splitlines() if line.strip()]
    assert len(cases) >= 20, "the shared corpus should cover all four outcomes generously"
    return cases


class CorpusBuildError(Exception):
    """A corpus row this loader cannot build (T-0027 criterion 12).

    Deliberately *not* an ``AssertionError``: ``tests/conftest.py`` turns a pending row into
    ``xfail(raises=(NotImplementedError, AssertionError))``, so a builder bug in a pending row
    would hide as an expected failure if it arrived as an assertion.
    """


def build_raw(spec: Any, name: str = "?") -> str:
    """Build a corpus payload from a ``build`` spec (tests/fixtures/hud/MANIFEST.md).

    The contract, implemented once per language (the JS half is ``buildRaw`` in
    ``tests/hud/protocol.test.mjs``): ``prefix + char * count + suffix``, where exactly one of
    ``count`` / ``fill_to_bytes`` gives the repeat count. With ``fill_to_bytes`` the count is
    ``(fill_to_bytes + delta_bytes - utf8len(prefix) - utf8len(suffix)) / utf8len(char)`` and must
    divide exactly, and ``fill_to_bytes`` must be ``codec.MAX_MESSAGE_BYTES`` — if the constant
    ever moves, the boundary rows fail instead of quietly testing the wrong boundary.

    Anything else raises (criterion 12): a row nobody can build must fail loudly rather than
    degenerate into a payload ``decode`` happens to call malformed.
    """
    if not isinstance(spec, dict):
        raise CorpusBuildError(f"corpus case {name!r}: 'build' must be an object, got {spec!r}")
    known = {"prefix", "char", "suffix", "count", "fill_to_bytes", "delta_bytes"}
    unknown = sorted(set(spec) - known)
    if unknown:
        raise CorpusBuildError(
            f"corpus case {name!r}: unknown build key(s) {unknown}; the builder contract in "
            f"tests/fixtures/hud/MANIFEST.md allows {sorted(known)}"
        )
    missing = sorted({"prefix", "char", "suffix"} - set(spec))
    if missing:
        raise CorpusBuildError(f"corpus case {name!r}: build spec is missing {missing}")
    prefix, char, suffix = str(spec["prefix"]), str(spec["char"]), str(spec["suffix"])
    if not char:
        raise CorpusBuildError(f"corpus case {name!r}: 'char' must be non-empty")
    if ("count" in spec) == ("fill_to_bytes" in spec):
        raise CorpusBuildError(
            f"corpus case {name!r}: exactly one of 'count' / 'fill_to_bytes' must be present"
        )
    if "count" in spec:
        if "delta_bytes" in spec:
            raise CorpusBuildError(f"corpus case {name!r}: 'delta_bytes' needs 'fill_to_bytes'")
        count = int(spec["count"])
    else:
        fill = int(spec["fill_to_bytes"])
        if fill != MAX_MESSAGE_BYTES:
            raise CorpusBuildError(
                f"corpus case {name!r}: fill_to_bytes is {fill} but "
                f"codec.MAX_MESSAGE_BYTES is {MAX_MESSAGE_BYTES}; the boundary rows must move "
                "with the constant, not test some other boundary"
            )
        room = fill + int(spec.get("delta_bytes", 0))
        room -= len(prefix.encode("utf-8")) + len(suffix.encode("utf-8"))
        count, remainder = divmod(room, len(char.encode("utf-8")))
        if remainder or count < 0:
            raise CorpusBuildError(
                f"corpus case {name!r}: {room} bytes of room does not divide by "
                f"{len(char.encode('utf-8'))} bytes per char (the count must be exact, not floored)"
            )
    return prefix + char * count + suffix


def raw_of(case: dict[str, Any]) -> str:
    if case.get("oversized"):
        return (
            '{"type":"astra.transcript","turn_id":"t","text":"'
            + "x" * (MAX_MESSAGE_BYTES + 64)
            + '"}'
        )
    name = str(case.get("name", "?"))
    if "build" in case:
        return build_raw(case["build"], name)
    raw = case.get("raw")
    if not isinstance(raw, str):
        raise CorpusBuildError(
            f"corpus case {name!r} has neither a string 'raw' nor a 'build' spec (criterion 12)"
        )
    return raw


def corpus_raw(name: str) -> str:
    """The payload of one corpus row by name, for tests that need a single case."""
    for case in corpus_cases():
        if case["name"] == name:
            return raw_of(case)
    raise CorpusBuildError(f"no corpus case named {name!r} in {CORPUS}")


def corpus_params() -> list[Any]:
    """The corpus as parametrisation values; a ``"pending"`` key becomes the pending marker.

    ``corpus_cases()`` keeps returning plain dicts, because the other corpus tests iterate it.
    """
    return [
        pytest.param(
            case,
            id=str(case["name"]),
            marks=([pytest.mark.pending(task=str(case["pending"]))] if case.get("pending") else []),
        )
        for case in corpus_cases()
    ]


def outcome_of(raw: str) -> tuple[str, str | None]:
    d = decode(raw)
    if isinstance(d, Malformed):
        return "malformed", d.type
    if isinstance(d, Unknown):
        return "unknown", d.type
    if isinstance(d, RealtimeEvent):
        return "realtime", d.type
    return "astra", str(getattr(d, "type", None))


@pytest.mark.parametrize("case", corpus_params())
def test_every_shared_corpus_case_has_the_outcome_it_declares(case):
    """Criterion 2: the corpus ``tests/hud/protocol.test.mjs`` checks the page against is itself
    checked against the production codec, so neither side can drift into agreeing on a lie."""
    got_kind, got_type = outcome_of(raw_of(case))
    assert got_kind == case["expect"], (
        f"corpus case {case['name']!r} declares {case['expect']!r} but the real "
        f"local_astra.contracts.codec.decode says {got_kind!r}"
    )
    if case.get("type") is not None:
        assert got_type == case["type"], f"corpus case {case['name']!r} type mismatch: {got_type!r}"


def test_the_corpus_covers_all_four_decode_outcomes():
    """Criterion 2: "for every inbound frame the page reaches one of four outcomes"."""
    kinds = {c["expect"] for c in corpus_cases()}
    assert kinds == {"astra", "unknown", "realtime", "malformed"}


def test_a_corpus_case_with_an_unbuildable_spec_is_an_error_not_a_malformed_row():
    """T-0027 criterion 12, Python half.

    ``decode`` of an empty/absent payload is ``Malformed``, so a row whose builder key nobody
    understands would go green on any ``"expect": "malformed"`` row *for the wrong reason*. The
    loader therefore refuses to build it at all.
    """
    for broken in (
        {"name": "unknown_key", "build": {"nonsense": 1}, "expect": "malformed"},
        {"name": "no_raw_no_build", "expect": "malformed"},
        {"name": "both_counts", "build": {**BOUNDARY_SPEC, "count": 3}, "expect": "malformed"},
        {"name": "neither_count", "build": {"prefix": "a", "char": "x", "suffix": "b"}},
        {"name": "not_an_object", "build": 7},
    ):
        with pytest.raises(CorpusBuildError):
            raw_of(broken)


def test_the_builder_refuses_a_count_that_does_not_divide_exactly():
    """The builder contract says "must divide exactly (assert, do not floor)": a floored count
    would build a payload one or two bytes short of the boundary it claims to sit on, which is
    exactly the mistake criteria 4a/4b exist to catch."""
    # "€" is 3 bytes; 8_388_608 - 49 - 2 = 8_388_557 bytes of room, and 8_388_557 % 3 == 2.
    assert (MAX_MESSAGE_BYTES - len(PREFIX.encode()) - len(SUFFIX.encode())) % 3 == 2
    with pytest.raises(CorpusBuildError):
        raw_of({"name": "euro_boundary", "build": {**BOUNDARY_SPEC, "char": "\u20ac"}})


def test_the_builder_pins_fill_to_bytes_to_the_codec_constant():
    """So that a change to ``MAX_MESSAGE_BYTES`` fails the boundary rows instead of silently
    moving them onto a boundary that is no longer the codec's."""
    with pytest.raises(CorpusBuildError):
        raw_of(
            {"name": "stale", "build": {**BOUNDARY_SPEC, "fill_to_bytes": MAX_MESSAGE_BYTES // 2}}
        )


def test_the_corpus_covers_the_byte_boundary_in_both_directions():
    """T-0027 criteria 1, 2 and 4: the shared corpus carries a payload accepted at exactly
    ``MAX_MESSAGE_BYTES``, one refused at one byte more, and a non-ASCII payload refused on its
    bytes — so deleting a row later is visible here rather than silently losing the coverage.

    The byte counts are hand arithmetic, not a second copy of the builder: the prefix is 49 bytes
    and the suffix 2, so 8_388_557 ``x`` fill the envelope exactly; "€" is 3 UTF-8 bytes, so
    3_000_000 of them plus 51 is 9_000_051; "😀" is 4, so 2_500_000 of them plus 51 is 10_000_051.
    """
    sized = {
        c["name"]: (len(raw_of(c).encode("utf-8")), c["expect"], raw_of(c).isascii())
        for c in corpus_cases()
        # T-0180 A2: the ``thought_`` build rows sit on the 16000-code-point thought cap, not on the
        # byte limit; their counts are pinned in tests/contract/test_t0180_thought_wire.py.
        if "build" in c and not str(c["name"]).startswith("thought_")
    }
    assert sized == {
        "multibyte_oversized_euro": (9_000_051, "malformed", False),
        "astral_oversized_defeats_a_times_three_shortcut": (10_000_051, "malformed", False),
        "ascii_exactly_at_the_limit": (8_388_608, "astra", True),
        "ascii_one_byte_over_the_limit": (8_388_609, "malformed", True),
    }
    accepted_at_limit = [n for n, (b, e, _) in sized.items() if b == MAX_MESSAGE_BYTES]
    refused_just_over = [n for n, (b, e, _) in sized.items() if b == MAX_MESSAGE_BYTES + 1]
    refused_multibyte = [
        n for n, (b, e, a) in sized.items() if b > MAX_MESSAGE_BYTES and not a and e == "malformed"
    ]
    assert accepted_at_limit == ["ascii_exactly_at_the_limit"]
    assert refused_just_over == ["ascii_one_byte_over_the_limit"]
    assert len(refused_multibyte) == 2, refused_multibyte


# --- T-0025: the assist panel's own constants ------------------------------------------------
#
# The panel declares its own client->server whitelist rather than widening ``protocol.js``
# ``CLIENT_TYPES`` (T-0025 criterion 12, and the test above stays unmodified). These parse
# ``hud/src/assists/gate.js`` the same way the rest of this module parses the shell: by source
# text, because ``hud/`` is buildless and the fast suite has no node dependency.

ASSISTS_DIR = REPO / "hud" / "src" / "assists"
GATE_JS = ASSISTS_DIR / "gate.js"
PANEL_JS_FILES = ("gate.js", "index.js", "extract.js")


def read_panel_module(path: Path, criterion: str) -> str:
    assert path.exists(), (
        f"T-0025 {criterion} adds {path.relative_to(REPO)} — the assist panel (docs/tasks/"
        "T-0025.md: the checkboxes, the badge and the egress gate)"
    )
    return path.read_text()


def string_list(src: str, name: str, path: Path) -> list[str]:
    """Read the string literals out of ``name = [...]`` **in order** (order is a criterion)."""
    m = re.search(rf"\b{name}\s*=\s*(?:new\s+Set\s*\()?\s*\[(.*?)\]", src, re.DOTALL)
    assert m is not None, (
        f"T-0025: {path.name} must define {name} as an array of string literals so this test can "
        "compare it with local_astra/contracts/"
    )
    return re.findall(r"""["']([^"']+)["']""", m.group(1))


def test_the_assist_panel_declares_its_own_client_to_server_whitelist():
    """Criterion 10: ``ASSIST_CLIENT_TYPES`` is exactly the two types and is a subset of the
    contract's ``CLIENT_TO_SERVER``; criterion 12: ``protocol.js`` ``CLIENT_TYPES`` is *not*
    widened to hold them (pinned, unmodified, by the test above)."""
    src = read_panel_module(GATE_JS, "criterion 10")
    declared = set(string_list(src, "ASSIST_CLIENT_TYPES", GATE_JS))
    assert declared == {"astra.assist.toggle", "astra.frame.text"}, (
        "criterion 10: the panel emits astra.assist.toggle and astra.frame.text and nothing "
        f"else — astra.ui.action is T-0010's and stays forbidden here; got {sorted(declared)}"
    )
    assert declared <= set(CLIENT_TO_SERVER), (
        "criterion 10: every type the panel emits is a client->server type in "
        "local_astra/contracts/protocol.py CLIENT_TO_SERVER"
    )


def test_the_assist_panel_kind_list_is_assist_flags_declaration_order():
    """Criterion 2: one row per kind, in ``AssistFlags`` field order (T-0012 amendment A3).

    Order, not membership: ``AssistState.texts_for`` returns the kinds in this order and the HUD
    rows mirror it, so a reversed list must fail here rather than showing the user a different
    order from the one the prompt suffix uses.
    """
    src = read_panel_module(GATE_JS, "criterion 2")
    kinds = string_list(src, "ASSIST_KINDS", GATE_JS)
    expected = list(AssistFlags.model_fields)
    assert kinds == expected, (
        "criterion 2: hud/src/assists/gate.js ASSIST_KINDS must equal "
        f"tuple(AssistFlags.model_fields) in order, which is {expected}; got {kinds}"
    )
    assert kinds == list(assists.ASSIST_KINDS), (
        "and it is the same order local_astra/perception/assists.ASSIST_KINDS uses"
    )


def test_the_assist_panel_defines_no_rate_constant_of_its_own():
    """Criterion 7: the rate limit is ``1000 / MAX_FPS`` with ``MAX_FPS`` *imported from*
    ``hud/src/frames.js`` — "the panel defines no second rate constant", because a second copy
    would drift from ``perception/policy.MAX_FPS`` without anything failing."""
    src = read_panel_module(GATE_JS, "criterion 7")
    assert re.search(
        r"""import\s*\{[^}]*\bMAX_FPS\b[^}]*\}\s*from\s*["']\.\./frames\.js["']""", src
    ), (
        "criterion 7: gate.js must import MAX_FPS from ../frames.js (the value "
        "tests/unit/test_hud_constants_parity.py already pins against perception/policy.MAX_FPS)"
    )
    for name in PANEL_JS_FILES:
        path = ASSISTS_DIR / name
        if not path.exists():
            continue
        body = re.sub(r"/\*.*?\*/", " ", path.read_text(), flags=re.DOTALL)
        body = re.sub(r"(?<!:)//[^\n]*", " ", body)
        # A *second* rate value: an fps-named identifier bound to a literal. A default parameter
        # (``maxFps = MAX_FPS``) is not one, and neither is ``1000 / maxFps``.
        offenders = re.findall(r"\b(?:MAX_FPS|ASSIST_FPS|FPS|fps)\s*=\s*-?\d", body)
        assert offenders == [], (
            f"criterion 7: hud/src/assists/{name} defines its own rate constant ({offenders}); "
            "the limit must be derived from the imported MAX_FPS"
        )
        # ... and a hard-coded window (``minGapMs = 1000``) is the same defect wearing a hat.
        for gap, rhs in re.findall(r"\b([A-Za-z_]*[Gg]ap[A-Za-z_]*)\s*=\s*([^;\n]+)", body):
            assert re.search(r"fps", rhs, re.IGNORECASE), (
                f"criterion 7: hud/src/assists/{name} sets {gap} to {rhs.strip()!r}, which does "
                "not derive from the imported MAX_FPS — the rate limit must move with "
                "perception/policy.MAX_FPS, not beside it"
            )


def test_the_assist_panel_text_cap_is_thirty_two_kibibytes():
    """Criterion 9: ``MAX_TEXT_BYTES`` is a panel constant of 32 KiB, counted in UTF-8 bytes.

    There is no Python counterpart to compare it with — ``AstraFrameText.text`` is unbounded and
    ``codec.MAX_MESSAGE_BYTES`` is 8 MiB — so the number comes from the task, and it is pinned
    here so that changing it is a visible edit rather than a quiet one.
    """
    src = read_panel_module(GATE_JS, "criterion 9")
    assert number(src, "MAX_TEXT_BYTES", GATE_JS) == 32 * 1024
    assert number(src, "MAX_TEXT_BYTES", GATE_JS) < MAX_MESSAGE_BYTES, (
        "criterion 9: the cap exists so one assist text cannot fill the codec's 8 MiB envelope"
    )


def test_the_assist_panel_never_derives_the_badge_from_the_flags():
    """Criterion 4: ``AssistFlags.badge()`` lives in Python and there is exactly one derivation
    of it (T-0012 criterion 8). The badge spellings ``dom``, ``pdf`` and ``a11y`` are not field
    names, so a literal of one under ``hud/src/assists/`` is a second derivation being built.
    ``cells`` is excluded from the scan on purpose: it is a *kind* name as well as a badge
    spelling, and building ``pixels+cells`` still needs ``pixels+``.
    """
    assert ASSISTS_DIR.is_dir(), "T-0025 adds hud/src/assists/"
    spellings = ('"dom"', "'dom'", '"pdf"', "'pdf'", '"a11y"', "'a11y'", "pixels+")
    offenders = []
    for path in sorted(ASSISTS_DIR.rglob("*.js")):
        body = re.sub(r"/\*.*?\*/", " ", path.read_text(), flags=re.DOTALL)
        body = re.sub(r"(?<!:)//[^\n]*", " ", body)
        offenders += [f"{path.relative_to(REPO)}: {s}" for s in spellings if s in body]
    assert offenders == [], (
        "criterion 4: the panel renders the badge the server sent, verbatim; it contains no code "
        f"that builds one from flags. Found {offenders}"
    )


# --- T-0037: the replay marker exists on both sides ---------------------------------------------

#: The five models docs/tasks/T-0037.md criterion 2 marks, and the wire types they encode to.
REPLAY_TYPES = {
    "AstraTranscript": "astra.transcript",
    "AstraSpoken": "astra.spoken",
    "AstraUiSurface": "astra.ui.surface",
    "AstraSpokenPlayed": "astra.spoken.played",
    "AstraTurnState": "astra.turn.state",
}


def model_table(src: str, wire_type: str) -> str:
    """The field table of one message in ``hud/src/protocol.js``'s ``MODELS``.

    Read as source text rather than imported, for the reason this whole module exists: ``hud/`` is
    buildless ES modules and the fast suite has no node. The slice runs from this type's key to
    the next ``"astra.…":`` key, which is where the object literal for the next message begins.
    """
    start = re.search(rf'^\s*"{re.escape(wire_type)}":\s*\[', src, re.MULTILINE)
    assert start is not None, (
        f'T-0008 criterion 2: hud/src/protocol.js MODELS must have an entry for "{wire_type}"'
    )
    rest = src[start.end() :]
    nxt = re.search(r'^\s{2}"astra\.[a-z.]+":\s*\[', rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


@pytest.mark.parametrize(("model_name", "wire_type"), sorted(REPLAY_TYPES.items()))
def test_the_replay_field_is_on_both_the_python_model_and_the_page(model_name, wire_type):
    """T-0037 criteria 2 and 10: the parity pin that makes the landmine impossible to forget.

    ``contracts/base.Contract`` forbids unknown fields and ``hud/src/protocol.js`` mirrors that
    literally, per key. A ``replay`` added to Python without the mirror does not degrade
    gracefully: **every** replayed message becomes ``kind: "malformed"`` on the page — counted,
    ignored, invisible — and the reloading page the task exists to fix stays empty while the
    server believes it caught it up. That is why group B is not optional.
    """
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, model_name)
    assert "replay" in model.model_fields, (
        f"T-0037 criterion 2: {model_name} gains `replay: bool = False` "
        "(local_astra/contracts/protocol.py)"
    )

    src = read_page_module(PROTOCOL_JS, "T-0037 criterion 10")
    table = model_table(src, wire_type)
    assert re.search(r"\breplay\s*:\s*bool\s*\(", table), (
        f"T-0037 criterion 10: hud/src/protocol.js's field table for {wire_type} must declare "
        "`replay: bool()` (optional — a live message omits it). Without it the page answers every "
        f"replayed {wire_type} with kind:'malformed'. The table reads:\n{table.strip()[:400]}"
    )
    assert "required" not in table.split("replay")[1].split("\n")[0], (
        "T-0037 criterion 2: replay has a default on the Python side, so the page must not "
        "require it either — a live message may omit the key entirely"
    )


# --- T-0039: the ask panel's own constants (criteria 11 and 12) ---------------------------------
#
# Same arrangement as T-0025's assist panel above: the panel declares its own client->server
# whitelist and ``protocol.js`` ``CLIENT_TYPES`` stays T-0008's two entries (pinned, unmodified,
# by ``test_the_page_emits_only_hello_and_frame_and_they_are_client_to_server_types``). Parsed by
# source text because ``hud/`` is buildless and the fast suite has no node dependency.

ASK_DIR = REPO / "hud" / "src" / "ask"
ASK_FORM_JS = ASK_DIR / "form.js"


def test_the_ask_panel_declares_its_own_client_to_server_whitelist():
    """Criterion 11: ``ASK_CLIENT_TYPES`` is exactly ``astra.text`` and the contract allows it."""
    assert ASK_FORM_JS.exists(), (
        f"T-0039 criterion 10 adds {ASK_FORM_JS.relative_to(REPO)} — the typed-question panel"
    )
    declared = string_list(ASK_FORM_JS.read_text(), "ASK_CLIENT_TYPES", ASK_FORM_JS)
    assert declared == ["astra.text"], (
        f"criterion 11: the panel emits astra.text and nothing else; it declares {declared}"
    )
    assert set(declared) <= set(CLIENT_TO_SERVER), (
        "criterion 11: astra.text is a client->server type in contracts/protocol.py "
        f"CLIENT_TO_SERVER (criterion 1), which lists {sorted(CLIENT_TO_SERVER)}"
    )
    assert "astra.text" not in set(SERVER_TO_CLIENT), (
        "criterion 1: client->server only — the two sets are disjoint and the page can never be "
        "sent one"
    )


def test_the_pages_client_whitelist_and_server_set_are_untouched_by_the_ask_panel():
    """Criterion 11/20: ``protocol.js`` ``CLIENT_TYPES`` and ``SERVER_TO_CLIENT`` have zero diff.

    The panel widening the shell's whitelist is how the per-owner scan stops meaning anything;
    T-0025 settled that and T-0039 follows it.
    """
    src = read_page_module(PROTOCOL_JS, "criterion 11")
    assert string_set(src, "CLIENT_TYPES", PROTOCOL_JS) == {"astra.hello", "astra.frame"}, (
        "criterion 11: protocol.js CLIENT_TYPES stays T-0008's two entries; astra.text belongs "
        "to hud/src/ask/form.js ASK_CLIENT_TYPES"
    )
    assert string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS) == set(SERVER_TO_CLIENT), (
        "criterion 8/20: the answer reaches the page over the existing server->client types — no "
        "new type and no new field, so this set is unchanged"
    )


def test_the_pages_text_cap_is_the_caps_the_contract_enforces():
    """Criterion 12: the client-side cap **is** the model's, so ``malformed`` is a backstop.

    The oracle is the contract itself: 2000 code points is accepted and 2001 is not, whatever
    number the page carries. A page whose cap drifted below the model's silently refuses legal
    questions; above it, the user gets an error instead of an answer.
    """
    assert ASK_FORM_JS.exists(), "T-0039 criterion 10 adds hud/src/ask/form.js"
    cap = int(number(ASK_FORM_JS.read_text(), "MAX_TEXT_CHARS", ASK_FORM_JS))

    from pydantic import ValidationError

    from local_astra.contracts import protocol

    assert hasattr(protocol, "AstraText"), (
        "T-0039 criterion 1: contracts/protocol.py gains AstraText(type='astra.text', text=str)"
    )
    AstraText = protocol.AstraText

    AstraText(text="q" * cap)  # the page's own cap must be a legal message
    with pytest.raises(ValidationError):
        AstraText(text="q" * (cap + 1))
    AstraText(text="\U0001f600" * cap)  # code points, not UTF-16 units and not bytes


# --- T-0043 group B: the session-log pair, the spec mirror's newest landmine ---------------------
#
# Criterion 20 of docs/tasks/T-0043.md: ``hud/src/protocol.js`` gains ``astra.log.state`` in
# ``SERVER_TO_CLIENT`` **with its field table**, and ``hud/src/log/gate.js`` declares
# ``LOG_CLIENT_TYPES = ["astra.log.toggle"]`` as its own egress, exactly as T-0025's and T-0039's
# panels do. ``CLIENT_TYPES`` stays T-0008's two entries (pinned, unmodified, by
# ``test_the_page_emits_only_hello_and_frame_and_they_are_client_to_server_types`` above and by
# ``test_the_pages_client_whitelist_and_server_set_are_untouched_by_the_ask_panel``; both of those
# compare ``SERVER_TO_CLIENT`` against the contract as a *set*, so they move with this task by
# construction and their T-0008/T-0039 scope statements are unchanged — see T-0043 criterion 20).
#
# Why the field table is asserted in **both** directions and not just for membership: the two
# sides are strict in opposite ways. A field the contract has and the page's table lacks makes
# every live ``astra.log.state`` ``kind: "malformed"`` on the page — counted, ignored, invisible —
# so the checkbox never updates and nothing anywhere reports an error (the exact shape of the
# T-0037 landmine the ``replay`` census above exists for). A field the page's table has and the
# contract does not is the mirror image: the page declares a key the server can never send, and
# whoever reads the page believes it is being sent one.

LOG_TOGGLE_TYPE = "astra.log.toggle"
LOG_STATE_TYPE = "astra.log.state"
LOG_DIR = REPO / "hud" / "src" / "log"
LOG_GATE_JS = LOG_DIR / "gate.js"

#: A field entry in one of ``hud/src/protocol.js``'s ``MODELS`` tables: ``name: kind(...)``. The
#: tables are written both on one line (``astra.error``) and one field per line, so the boundary
#: is a ``{``/``,``/line start rather than an indent.
JS_FIELD = re.compile(
    r"(?:[{,]|^)\s*([a-z_][a-z_0-9]*)\s*:\s*(?:str|int|bool|num|list|obj|oneOf|model)\s*\(",
    re.MULTILINE,
)


def js_field_names(table: str) -> list[str]:
    return JS_FIELD.findall(table)


def log_state_model():
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, "AstraLogState", None)
    assert model is not None, (
        "T-0043 criterion 16: contracts/protocol.py gains AstraLogState(type='astra.log.state', "
        "on, path, records, dropped, bytes, error) — the Python side is the spec and the page "
        "mirrors it (docs/tasks/T-0043.md)"
    )
    return model


def test_the_log_state_is_in_both_server_to_client_lists():
    """Criterion 20: the type is on both sides, named rather than only set-compared.

    The set-equality rows above are satisfied by *both* lists forgetting it; this one is not.
    """
    log_state_model()
    src = read_page_module(PROTOCOL_JS, "T-0043 criterion 20")
    assert LOG_STATE_TYPE in set(SERVER_TO_CLIENT), (
        "T-0043 criterion 16: contracts/protocol.py SERVER_TO_CLIENT gains astra.log.state; it "
        f"holds {sorted(SERVER_TO_CLIENT)}"
    )
    assert LOG_STATE_TYPE in string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS), (
        "T-0043 criterion 20: hud/src/protocol.js mirrors it, or the page drops the one message "
        "that tells it whether this session is being recorded"
    )
    assert LOG_STATE_TYPE not in set(CLIENT_TO_SERVER), (
        "criterion 16: server→client only; a page that sends one is refused as wrong_direction"
    )
    assert LOG_TOGGLE_TYPE in set(CLIENT_TO_SERVER), (
        f"criterion 16: astra.log.toggle is client→server; CLIENT_TO_SERVER is "
        f"{sorted(CLIENT_TO_SERVER)}"
    )
    assert LOG_TOGGLE_TYPE not in string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS), (
        "criterion 20: the page dispatches the state, never the toggle — SERVER_TO_CLIENT is the "
        "page's inbound table and the toggle is its own egress"
    )
    assert string_set(src, "CLIENT_TYPES", PROTOCOL_JS) == {"astra.hello", "astra.frame"}, (
        "criterion 20: CLIENT_TYPES stays T-0008's two entries — astra.log.toggle belongs to "
        "hud/src/log/gate.js LOG_CLIENT_TYPES, as astra.text belongs to the ask panel's"
    )


def test_the_pages_field_table_for_the_log_state_matches_the_contract_field_for_field():
    """Criterion 20, both directions: same fields, same optionality, same nullability.

    The oracle is ``AstraLogState.model_fields`` — the contract — never a second list written
    here. ``path`` and ``error`` are nullable in Python, so the page must declare them nullable
    too: a page that requires a string for ``path`` answers "logging is off" (``path: null``)
    with ``malformed``, which is precisely the state the user most needs to see.
    """
    model = log_state_model()
    src = read_page_module(PROTOCOL_JS, "T-0043 criterion 20")
    table = model_table(src, LOG_STATE_TYPE)
    declared = js_field_names(table)

    assert set(declared) == set(model.model_fields) - {"type"}, (
        "criterion 20: the page's field table and the contract's fields are the same set — a "
        "field on one side and not the other turns every live astra.log.state into a malformed "
        f"one on the page. Python has {sorted(set(model.model_fields) - {'type'})}, the page has "
        f"{sorted(set(declared))}. The table reads:\n{table.strip()[:400]}"
    )
    assert len(declared) == len(set(declared)), f"…each declared once; got {declared}"

    for name, field in model.model_fields.items():
        if name == "type":
            continue
        entry = re.search(rf"\b{name}\s*:\s*[a-z]+\(([^)]*)\)", table)
        assert entry is not None, f"criterion 20: no entry for {name} in the page's table"
        options = entry.group(1)
        nullable_in_python = type(None) in typing.get_args(field.annotation)
        assert ("nullable" in options) == nullable_in_python, (
            f"criterion 20: {name} is {'' if nullable_in_python else 'not '}nullable in "
            f"contracts/protocol.py AstraLogState and the page says {options.strip()!r}; a page "
            "stricter than the contract answers a legal message with malformed, and a page "
            "looser than it accepts one the server can never send"
        )
        assert ("required" in options) == field.is_required(), (
            f"criterion 20: {name} is {'' if field.is_required() else 'not '}required in the "
            f"contract and the page says {options.strip()!r}"
        )


def test_the_log_panel_declares_its_own_client_to_server_whitelist():
    """Criterion 19/20: ``LOG_CLIENT_TYPES`` is exactly the toggle, and the contract allows it."""
    assert LOG_GATE_JS.exists(), (
        f"T-0043 criterion 19 adds {LOG_GATE_JS.relative_to(REPO)} — the log panel's DOM-free half"
    )
    declared = string_list(LOG_GATE_JS.read_text(), "LOG_CLIENT_TYPES", LOG_GATE_JS)
    assert declared == [LOG_TOGGLE_TYPE], (
        f"criterion 19: the panel emits astra.log.toggle and nothing else; it declares {declared}"
    )
    assert set(declared) <= set(CLIENT_TO_SERVER), (
        "criterion 19: every type the panel emits is a client→server type in "
        f"contracts/protocol.py CLIENT_TO_SERVER, which lists {sorted(CLIENT_TO_SERVER)}"
    )


# --- T-0054: the fence notice, and why it is a new *type* ---------------------------------------
#
# Criterion 13 of docs/tasks/T-0054.md: ``hud/src/protocol.js`` gains ``astra.ui.fence`` in
# ``SERVER_TO_CLIENT`` **with its full field table**. Criterion 8 is the Python half. The row
# below named ``…_is_a_new_type_and_not_a_field_on_the_surface_message`` is the one the shape was
# chosen for, and the reason is a measured asymmetry in this very decoder (T-0054 D1):
#
#   * a **field** added to an existing message in Python and not mirrored here turns *every*
#     message of that type into ``kind: "malformed"`` — the shown pane goes blank on every turn
#     until the page catches up (T-0037's landmine, T-0047 criterion 17's warning);
#   * a **type** added in Python and not mirrored decodes as ``kind: "unknown"`` and is ignored
#     and counted (``protocol.js``'s header, four outcomes).
#
# So the new-type shape degrades gracefully and the new-field shape is a page-killer, which is
# why groups A and B of that task are safe to implement concurrently and why nothing may be added
# to ``astra.ui.surface``. ``test_the_pages_server_to_client_set_equals_the_contract`` above
# compares the two lists as *sets* and therefore moves with this task by construction — it is
# named in T-0054's Owned files as a pin the task legitimately moves, and it is also why groups A
# and B merge together.

UI_FENCE_TYPE = "astra.ui.fence"
UI_SURFACE_TYPE = "astra.ui.surface"
FENCE_JS = REPO / "hud" / "src" / "ui" / "fence.js"


#: ``contracts/llm.FenceClose``, read from the contract rather than spelled here.
def fence_close_values() -> set[str]:
    import typing

    from local_astra.contracts.llm import FenceClose

    return set(typing.get_args(FenceClose))


def ui_fence_model():
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, "AstraUiFence", None)
    assert model is not None, (
        "T-0054 criterion 8: contracts/protocol.py gains AstraUiFence(type='astra.ui.fence', "
        "turn_id, canvases, close=None, continued=False, speech_refused=0) — the Python side is "
        "the spec and the page mirrors it (docs/tasks/T-0054.md)"
    )
    return model


def test_the_ui_fence_is_in_both_server_to_client_lists():
    """Criterion 8/13: the type is on both sides, named rather than only set-compared.

    The set-equality row above is satisfied by *both* lists forgetting it; this one is not.
    """
    ui_fence_model()
    src = read_page_module(PROTOCOL_JS, "T-0054 criterion 13")
    assert UI_FENCE_TYPE in set(SERVER_TO_CLIENT), (
        "T-0054 criterion 8: contracts/protocol.py SERVER_TO_CLIENT gains astra.ui.fence; it "
        f"holds {sorted(SERVER_TO_CLIENT)}"
    )
    assert UI_FENCE_TYPE in string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS), (
        "T-0054 criterion 13: hud/src/protocol.js mirrors it, or the page drops the one message "
        "that says whether the interface it is showing is whole, cut off or abandoned — and the "
        "one that says a refusal fired"
    )
    assert UI_FENCE_TYPE not in set(CLIENT_TO_SERVER), (
        "criterion 8: server→client only; a page that sends one is refused as wrong_direction"
    )
    assert string_set(src, "CLIENT_TYPES", PROTOCOL_JS) == {"astra.hello", "astra.frame"}, (
        "criterion 13: CLIENT_TYPES stays T-0008's two entries — this panel sends nothing at all "
        "(criterion 17: no client→server message is added by the fence panel)"
    )


def test_the_ui_fence_is_a_new_type_and_not_a_field_on_the_surface_message():
    """Criterion 8/12 and D1 — the row that goes red if an implementer takes the other shape.

    "No new field on any existing wire message" (the Goal's non-goals), and criterion 12 names
    ``astra.ui.surface`` among the things that do not change. An implementer who adds
    ``fence_close`` to ``AstraUiSurface`` instead of adding a type would satisfy "the page can
    tell the three states apart" and would blank the shown pane on every turn until
    ``protocol.js`` caught up — on a *merged* server with an *unmerged* page, which is precisely
    the window groups A and B are allowed to have.

    Both sides are checked: the contract's own field set, and the page's table.
    """
    import local_astra.contracts.protocol as protocol

    surface = protocol.AstraUiSurface
    assert set(surface.model_fields) - {"type"} == {
        "surface_id",
        "canvas_idx",
        "final",
        "messages",
        "repaired",
        "replay",
    }, (
        "T-0054 criterion 12: astra.ui.surface is unchanged by this task — the fence lifecycle "
        "is a new message type, never a field here (D1). AstraUiSurface now has "
        f"{sorted(set(surface.model_fields) - {'type'})}"
    )
    src = read_page_module(PROTOCOL_JS, "T-0054 criterion 13")
    surface_table = model_table(src, UI_SURFACE_TYPE)
    # ``messages`` is declared ``list({ item: obj(), required: true })`` and ``JS_FIELD`` is a
    # flat scanner, so the nested ``item`` spec looks like a field name to it. It is the *list
    # element's* spec, not a message field, and it is the only nested one in this table.
    declared = {name for name in js_field_names(surface_table) if name != "item"}
    assert declared == set(surface.model_fields) - {"type"}, (
        "criterion 12/13: and the page's table for astra.ui.surface is unchanged too. Python has "
        f"{sorted(set(surface.model_fields) - {'type'})}, the page has "
        f"{sorted(declared)}"
    )
    assert UI_FENCE_TYPE in string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS), (
        "…and the lifecycle arrived as its own type in the dispatch table, which is the half of "
        "D1 that makes the unmirrored case survivable: an unknown *type* decodes as "
        "kind:'unknown' and is ignored and counted, while an unmirrored *field* makes every "
        "message of that type malformed and blanks the pane"
    )


def test_the_pages_field_table_for_the_fence_matches_the_contract_field_for_field():
    """Criterion 13, both directions: same fields, same optionality, same nullability.

    The oracle is ``AstraUiFence.model_fields`` — the contract — never a second list written
    here. A page stricter than the contract answers a legal message with ``malformed`` (so the
    pane never learns the surface was abandoned); a page looser than it declares a key the server
    can never send.
    """
    model = ui_fence_model()
    src = read_page_module(PROTOCOL_JS, "T-0054 criterion 13")
    table = model_table(src, UI_FENCE_TYPE)
    declared = js_field_names(table)

    assert set(declared) == set(model.model_fields) - {"type"}, (
        "criterion 13: the page's field table and the contract's fields are the same set. Python "
        f"has {sorted(set(model.model_fields) - {'type'})}, the page has {sorted(set(declared))}."
        f" The table reads:\n{table.strip()[:400]}"
    )
    assert len(declared) == len(set(declared)), f"…each declared once; got {declared}"

    for name, field in model.model_fields.items():
        if name == "type":
            continue
        entry = re.search(rf"\b{name}\s*:\s*[a-zA-Z]+\(([^)]*)\)", table)
        assert entry is not None, f"criterion 13: no entry for {name} in the page's table"
        options = entry.group(1)
        nullable_in_python = type(None) in typing.get_args(field.annotation)
        assert ("nullable" in options) == nullable_in_python, (
            f"criterion 13: {name} is {'' if nullable_in_python else 'not '}nullable in "
            f"contracts/protocol.py AstraUiFence and the page says {options.strip()!r}"
        )
        assert ("required" in options) == field.is_required(), (
            f"criterion 13: {name} is {'' if field.is_required() else 'not '}required in the "
            f"contract and the page says {options.strip()!r}"
        )


def test_the_pages_close_enum_is_the_contracts_four_fence_values():
    """Criterion 13: ``close`` is an enum of the four ``FenceClose`` values, not a free string.

    An unknown value must decode ``malformed`` rather than render, which is only true if the page
    declares the enum. A page that accepts any string renders a future ``close`` value as
    whatever its status table falls through to — the way a panel starts lying again.
    """
    ui_fence_model()
    src = read_page_module(PROTOCOL_JS, "T-0054 criterion 13")
    table = model_table(src, UI_FENCE_TYPE)
    entry = re.search(r"\bclose\s*:\s*oneOf\(([^)]*)\)", table)
    assert entry is not None, (
        "criterion 13: close is declared with oneOf(<the four FenceClose values>, {nullable:true})"
        f"; the page's table reads:\n{table.strip()[:400]}"
    )
    declared = set(re.findall(r"""["']([^"']+)["']""", entry.group(1)))
    if not declared:  # the values may be a named constant, e.g. FENCE_CLOSES
        named = re.search(r"oneOf\(\s*([A-Z_][A-Z_0-9]*)", entry.group(0))
        assert named is not None, (
            f"criterion 13: close's enum values are not readable: {entry.group(0)}"
        )
        declared = string_set(src, named.group(1), PROTOCOL_JS)
    assert declared == fence_close_values(), (
        "criterion 13: the page's close enum equals contracts/llm.FenceClose, which is "
        f"{sorted(fence_close_values())}; the page declares {sorted(declared)}"
    )


def test_the_fence_panel_is_its_own_module_and_declares_no_egress():
    """Criteria 14 and 17: ``hud/src/ui/fence.js`` exists, is pure, and sends nothing.

    A pure function of one decoded message: it holds no state, reads no clock and derives nothing
    the server did not send — in particular it never infers "complete" from the *absence* of a
    message (D5). This row is the textual half; the behavioural half is
    ``tests/hud/ui_fence.test.mjs``.
    """
    assert FENCE_JS.is_file(), (
        f"T-0054 criterion 14 adds hud/{FENCE_JS.relative_to(REPO / 'hud')} — describeFence("
        "message) -> { status, label, canvases, refused }, DOM-free and clock-free"
    )
    src = FENCE_JS.read_text()
    assert "export function describeFence" in src or "export const describeFence" in src, (
        "criterion 14: the module exports describeFence(message); tests/hud/ui_fence.test.mjs "
        "and tests/contract/test_fence_reaches_the_human.py both import that name"
    )
    for banned, why in (
        ("Date.now", "criterion 14: clock-free — it must not compute elapsed time"),
        ("performance.now", "criterion 14: clock-free"),
        ("document", "criterion 14: DOM-free; the DOM half is ui/index.js"),
        ("send(", "criterion 17: the panel sends no client→server message"),
    ):
        assert banned not in src, f"{why}; hud/src/ui/fence.js contains {banned!r}"
    assert UI_SURFACE_TYPE not in src, (
        "criterion 14/D5: describeFence is a pure function of one astra.ui.fence. A module that "
        "also watches surfaces is a module that accumulates, and an accumulating panel can "
        "contradict the server after a reconnect — this message is deliberately not replayed."
    )


# --- T-0090: astra.tool.call on both sides (criterion 2, the T-0043 pair pattern) ---------------
#
# docs/tasks/T-0090.md criterion 2: ``hud/src/protocol.js`` mirrors the field table of
# ``contracts/protocol.AstraToolCall`` and decodes it strictly. The oracle is the contract's own
# ``model_fields`` and its ``Literal`` arguments — never a second list written here — so the page
# cannot drift into agreeing with a copy. A field on one side only makes every live tool call
# ``kind: "malformed"`` on the page (the T-0037 landmine): the card would never appear and nothing
# would say why.

TOOL_CALL_TYPE = "astra.tool.call"


def tool_call_contract():
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, "AstraToolCall", None)
    assert model is not None, (
        "T-0090 criterion 1: contracts/protocol.py gains AstraToolCall(type='astra.tool.call', "
        "turn_id, call_id, name, arguments, status, source, action, reason, query_problem, "
        "replay=False) — the Python side is the spec and the page mirrors it"
    )
    return model


def literal_values(annotation: Any) -> set[str]:
    """The string literals of a ``Literal[...]`` annotation (``| None`` unwrapped)."""
    out: set[str] = set()
    for arg in typing.get_args(annotation) or (annotation,):
        if typing.get_origin(arg) is typing.Literal:
            out.update(str(v) for v in typing.get_args(arg))
    if typing.get_origin(annotation) is typing.Literal:
        out.update(str(v) for v in typing.get_args(annotation))
    return out


def test_the_tool_call_is_in_both_server_to_client_lists():
    """Criteria 1 and 2, named rather than only set-compared (both lists forgetting it is equal)."""
    tool_call_contract()
    src = read_page_module(PROTOCOL_JS, "T-0090 criterion 2")
    assert TOOL_CALL_TYPE in set(SERVER_TO_CLIENT), (
        f"T-0090 criterion 1: SERVER_TO_CLIENT gains {TOOL_CALL_TYPE}; it holds "
        f"{sorted(SERVER_TO_CLIENT)}"
    )
    assert TOOL_CALL_TYPE in string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS), (
        "T-0090 criterion 2: hud/src/protocol.js mirrors it, or the page counts every proposed "
        "action as an unknown type and the card never appears"
    )
    assert TOOL_CALL_TYPE not in set(CLIENT_TO_SERVER), "criterion 1: server→client only"
    assert string_set(src, "CLIENT_TYPES", PROTOCOL_JS) == {"astra.hello", "astra.frame"}, (
        "T-0090 implementation notes: the card is display-only and sends nothing back, so "
        "CLIENT_TYPES stays T-0008's two entries"
    )


def test_the_pages_field_table_for_the_tool_call_matches_the_contract_field_for_field():
    """Criterion 2, both directions: same fields, same optionality, same nullability."""
    model = tool_call_contract()
    src = read_page_module(PROTOCOL_JS, "T-0090 criterion 2")
    table = model_table(src, TOOL_CALL_TYPE)
    declared = js_field_names(table)
    python = set(model.model_fields) - {"type"}
    assert set(declared) == python, (
        "T-0090 criterion 2: the page's field table and the contract's fields are the same set. "
        f"Python has {sorted(python)}, the page has {sorted(set(declared))}. The table reads:\n"
        f"{table.strip()[:500]}"
    )
    assert len(declared) == len(set(declared)), f"…each declared once; got {declared}"
    for name, field in model.model_fields.items():
        if name == "type":
            continue
        entry = re.search(rf"\b{name}\s*:\s*[a-zA-Z]+\(([^)]*)\)", table)
        assert entry is not None, f"criterion 2: no entry for {name} in the page's table"
        options = entry.group(1)
        nullable_in_python = type(None) in typing.get_args(field.annotation)
        assert ("nullable" in options) == nullable_in_python, (
            f"criterion 2: {name} is {'' if nullable_in_python else 'not '}nullable in "
            f"AstraToolCall and the page says {options.strip()!r}"
        )
        assert ("required" in options) == field.is_required(), (
            f"criterion 2: {name} is {'' if field.is_required() else 'not '}required in the "
            f"contract and the page says {options.strip()!r}"
        )
    assert re.search(r"\barguments\s*:\s*obj\s*\(", table), (
        "criterion 1/2: arguments is a dict in the contract, so the page declares it obj() — a "
        f"str() or list() would refuse every legal call. The table reads:\n{table.strip()[:500]}"
    )
    assert re.search(r"\breplay\s*:\s*bool\s*\(", table), (
        "criterion 1/3: replay is a bool on the page too, or every journalled card replayed to a "
        "reloading page is malformed"
    )


@pytest.mark.parametrize("field_name", ["status", "source"])
def test_the_pages_tool_call_enums_are_the_contracts_literals(field_name):
    """Criterion 2: ``status`` and ``source`` are enums on the page, equal to the contract's."""
    model = tool_call_contract()
    expected = literal_values(model.model_fields[field_name].annotation)
    assert expected, (
        f"criterion 1: AstraToolCall.{field_name} is a Literal; its annotation is "
        f"{model.model_fields[field_name].annotation!r}"
    )
    src = read_page_module(PROTOCOL_JS, "T-0090 criterion 2")
    table = model_table(src, TOOL_CALL_TYPE)
    entry = re.search(rf"\b{field_name}\s*:\s*oneOf\(([^)]*)\)", table)
    assert entry is not None, (
        f"criterion 2: {field_name} is declared oneOf(<the contract's literals>, …) on the page so "
        f"an unknown value decodes malformed; the table reads:\n{table.strip()[:500]}"
    )
    declared = set(re.findall(r"""["']([^"']+)["']""", entry.group(1)))
    if not declared:  # the values may be a named constant
        named = re.search(r"oneOf\(\s*([A-Z_][A-Z_0-9]*)", entry.group(0))
        assert named is not None, f"criterion 2: unreadable enum: {entry.group(0)}"
        declared = string_set(src, named.group(1), PROTOCOL_JS)
    assert declared == expected, (
        f"criterion 2: the page's {field_name} enum equals the contract's {sorted(expected)}; the "
        f"page declares {sorted(declared)}"
    )


def test_the_contract_status_and_source_literals_are_the_criterion_one_lists():
    """Criterion 1, hand-transcribed: the six statuses and the two sources, nothing else."""
    model = tool_call_contract()
    assert literal_values(model.model_fields["status"].annotation) == {
        "proposed",
        "repairing",
        "dry_run",
        "executed",
        "refused",
        "cancelled",
    }
    assert literal_values(model.model_fields["source"].annotation) == {"draft", "repair"}


def test_the_corpus_carries_the_tool_call_rows_both_sides_read():
    """Criterion 2: "a byte-identical corpus row covers server and page" — at least one accepted
    and one refused ``astra.tool.call`` row, read by this module and by protocol.test.mjs."""
    rows = [c for c in corpus_cases() if TOOL_CALL_TYPE in str(c.get("raw", ""))]
    kinds = {c["expect"] for c in rows}
    assert {"astra", "malformed"} <= kinds, (
        "T-0090 criterion 2: tests/fixtures/hud/decode_corpus.jsonl carries astra.tool.call rows "
        f"that decode and rows that are refused; it has {[(c['name'], c['expect']) for c in rows]}"
    )


# --- T-0128c B1: astra.canvas.step mirrors AstraCanvasStep ----------------------------------------
#
# docs/tasks/T-0128c.md Finding 1: the page's table mirrors an old shape (accepted/entropy
# required, no canvas/stable/phase/committed_tokens, step < max_steps), so it refuses every real
# tap frame. The corpus rows tagged ``"pending_js": "T-0128c"`` are checked against the real codec
# by ``test_every_shared_corpus_case_has_the_outcome_it_declares`` above — ungated, because the
# Python side is the spec and already says what they are — and against the page by
# ``tests/hud/protocol.test.mjs`` through its T-0128c group B gate. The field-table rows below are
# the page's half and are pending until group B lands.

CANVAS_STEP_TYPE = "astra.canvas.step"

#: The rows docs/tasks/T-0128c.md *Tests to write* names (accepted, then malformed), plus the
#: guard's other sides. Hand-listed so deleting one is visible here.
T0128C_CORPUS = {
    "canvas_step_denoise_real_shape_cl8": "astra",
    "canvas_step_denoise_phase_omitted_defaults_to_denoise": "astra",
    "canvas_step_denoise_at_max_steps": "astra",
    "canvas_step_denoise_with_the_optional_fields": "astra",
    "canvas_step_commit_real_shape": "astra",
    "canvas_step_commit_at_step_zero": "astra",
    "canvas_step_commit_with_explicit_null_rows": "astra",
    "canvas_step_denoise_missing_stable": "malformed",
    "canvas_step_denoise_with_committed_ids": "malformed",
    "canvas_step_denoise_past_max_steps": "malformed",
    "canvas_step_stable_too_short": "malformed",
    "canvas_step_stable_odd_hex_digits": "malformed",
    "canvas_step_stable_not_hex": "malformed",
    "canvas_step_canvas_row_too_long": "malformed",
    "canvas_step_unknown_phase": "malformed",
    "canvas_step_commit_with_rows": "malformed",
    "canvas_step_commit_with_empty_ids": "malformed",
    "canvas_step_commit_past_max_steps": "malformed",
    "canvas_step_commit_ids_exceed_canvas_length": "malformed",
}


def _js_options(table: str, name: str) -> tuple[str, str] | None:
    """``(kind, options)`` of ``name: kind(options)`` in a page field table, parentheses balanced.

    The T-0090 rows' ``[^)]*`` stops at the first ``)``, which inside ``list({ item: int(), … })``
    is the item's; a list field's ``required``/``nullable`` would then be invisible.
    """
    m = re.search(rf"(?:[{{,]|^)\s*{name}\s*:\s*([a-zA-Z]+)\s*\(", table, re.MULTILINE)
    if m is None:
        return None
    depth, i = 1, m.end()
    while i < len(table) and depth:
        depth += {"(": 1, ")": -1}.get(table[i], 0)
        i += 1
    return m.group(1), table[m.end() : i - 1]


def test_the_corpus_carries_the_t0128c_canvas_step_rows_with_the_codecs_verdicts():
    """T-0128c B1: the shared rows exist and each declares what the real codec says (premise of
    both halves; the per-row verdicts are also checked by the parametrised corpus row above)."""
    rows = {c["name"]: c for c in corpus_cases() if c.get("pending_js") == "T-0128c"}
    assert {n: rows[n]["expect"] for n in rows} == T0128C_CORPUS, (
        "T-0128c B1: tests/fixtures/hud/decode_corpus.jsonl carries exactly the T-0128c canvas-step "
        f"rows this module lists; it has {sorted(rows)}"
    )
    for name, case in rows.items():
        assert outcome_of(raw_of(case)) == (case["expect"], CANVAS_STEP_TYPE), name
    old = next(c for c in corpus_cases() if c["name"] == "canvas_step_lengths_disagree")
    assert old["expect"] == "malformed" and "pending_js" not in old, (
        "T-0128c Tests to write: canvas_step_lengths_disagree stays malformed (and ungated)"
    )


def test_the_pages_field_table_for_the_canvas_step_matches_the_contract_field_for_field():
    """T-0128c B1: same fields, same optionality, same nullability as ``AstraCanvasStep``."""
    from local_astra.contracts.protocol import AstraCanvasStep

    src = read_page_module(PROTOCOL_JS, "T-0128c B1")
    table = model_table(src, CANVAS_STEP_TYPE)
    # ``item`` is a list spec's element type (``list({ item: int() })``), not a field.
    declared = [n for n in js_field_names(table) if n != "item"]
    python = set(AstraCanvasStep.model_fields) - {"type"}
    assert set(declared) == python, (
        "T-0128c B1: the page's astra.canvas.step table and AstraCanvasStep have the same fields. "
        f"Python has {sorted(python)}, the page has {sorted(set(declared))} (missing "
        f"{sorted(python - set(declared))}, extra {sorted(set(declared) - python)}). The table "
        f"reads:\n{table.strip()[:700]}"
    )
    assert len(declared) == len(set(declared)), f"…each declared once; got {declared}"
    problems = []
    for name, field in AstraCanvasStep.model_fields.items():
        if name == "type":
            continue
        entry = _js_options(table, name)
        if entry is None:
            problems.append(f"{name}: no entry")
            continue
        kind, options = entry
        # Only the field's own options: strip a nested item spec such as ``item: int()``.
        own = re.sub(r"item\s*:\s*[a-zA-Z]+\([^()]*\)", "", options)
        nullable = type(None) in typing.get_args(field.annotation)
        if ("nullable" in own) != nullable:
            problems.append(
                f"{name}: nullable is {nullable} in AstraCanvasStep, page says {options!r}"
            )
        if ("required" in own) != field.is_required():
            problems.append(
                f"{name}: required is {field.is_required()} in AstraCanvasStep, page says {options!r}"
            )
        if name in ("canvas", "argmax", "committed_tokens") and (
            kind != "list" or not re.search(r"item\s*:\s*int\s*\(", options)
        ):
            problems.append(f"{name}: a list of ints in the contract, page says {kind}({options})")
        if name == "entropy" and (kind != "list" or not re.search(r"item\s*:\s*num\s*\(", options)):
            problems.append(
                f"entropy: a list of floats in the contract, page says {kind}({options})"
            )
        if name in ("stable", "accepted") and "[0-9a-f]" not in options:
            problems.append(f"{name}: the contract's pattern ^[0-9a-f]*$, page says {options!r}")
    assert problems == [], "T-0128c B1, field by field:\n  " + "\n  ".join(problems)


def test_the_pages_canvas_step_phase_enum_is_the_contracts_literal():
    """T-0128c B1: ``phase`` ∈ {denoise, commit} on both sides, so a third value decodes malformed."""
    from local_astra.contracts.protocol import AstraCanvasStep

    expected = literal_values(AstraCanvasStep.model_fields["phase"].annotation)
    assert expected == {"denoise", "commit"}, f"premise: the contract's phase literal is {expected}"
    src = read_page_module(PROTOCOL_JS, "T-0128c B1")
    table = model_table(src, CANVAS_STEP_TYPE)
    entry = _js_options(table, "phase")
    assert entry is not None and entry[0] == "oneOf", (
        f"T-0128c B1: phase is declared oneOf([...]) on the page; the table reads:\n{table[:700]}"
    )
    declared = set(re.findall(r"""["']([^"']+)["']""", entry[1]))
    if not declared:
        named = re.search(r"^\s*([A-Z_][A-Z_0-9]*)", entry[1])
        assert named is not None, f"unreadable enum: {entry}"
        declared = string_set(src, named.group(1), PROTOCOL_JS)
    assert declared == expected, (
        f"T-0128c B1: page phase enum {sorted(declared)} vs {sorted(expected)}"
    )


# --- T-0180 A2: the thought cap is one number on both sides ----------------------------------------


def test_the_thought_cap_is_sixteen_thousand_code_points_on_both_sides():
    """T-0180 A2/A6 (H-T180-5): ``THOUGHT_MAX_CHARS = 16000`` in ``contracts/protocol.py`` and the
    page's ``hud/src/protocol.js`` defines the same plain number, so the two decoders refuse the same
    ``text``. 16000 is the task's literal, not read from either side."""
    from local_astra.contracts import protocol

    python = getattr(protocol, "THOUGHT_MAX_CHARS", None)
    assert python == 16000, (
        f"T-0180 A6: contracts/protocol.py defines THOUGHT_MAX_CHARS = 16000; found {python!r}"
    )
    src = read_page_module(PROTOCOL_JS, "T-0180 A2")
    assert re.search(r"\bTHOUGHT_MAX_CHARS\s*[:=]", src), (
        "T-0180 A2: hud/src/protocol.js defines THOUGHT_MAX_CHARS as a plain number (the parity "
        "row reads it from the source)"
    )
    assert number(src, "THOUGHT_MAX_CHARS", PROTOCOL_JS) == float(python), (
        "T-0180 A2: the page's THOUGHT_MAX_CHARS equals the contract's"
    )


# --- T-0213 A2r-1: the settings state ------------------------------------------------------------
#
# ``astra.settings.state`` is server -> client only and the page mirrors it: named on both lists (the set rows above
# are satisfied by both forgetting it) with a ``MODELS`` entry declaring every top-level field of the contract. The
# nested item/catalog/refusal fields also appear in that table's text, so the page's names are a superset check; the
# ``required`` flag is compared for the five scalar top-level fields A2r-1 names (a list field's options hold nested
# parentheses this row does not parse).

SETTINGS_STATE_TYPE = "astra.settings.state"
SETTINGS_STATE_SCALARS = ("rev", "system_sha12", "instructions", "instructions_max", "replay")


def test_the_settings_state_is_on_both_server_to_client_lists_with_its_fields():
    """T-0213 A2r-1: the type is in ``SERVER_TO_CLIENT`` on both sides, never client -> server, and the page's
    ``MODELS`` table declares every top-level field of ``AstraSettingsState``, the scalars with the contract's
    ``required``."""
    from local_astra.contracts import protocol

    model = getattr(protocol, "AstraSettingsState", None)
    assert model is not None, (
        "T-0213 A2r-1: contracts/protocol.py AstraSettingsState (astra.settings.state) does not exist yet"
    )
    src = read_page_module(PROTOCOL_JS, "T-0213 A2r-1")
    assert SETTINGS_STATE_TYPE in set(SERVER_TO_CLIENT), sorted(SERVER_TO_CLIENT)
    assert SETTINGS_STATE_TYPE in string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS), (
        "T-0213 A2r-1: hud/src/protocol.js SERVER_TO_CLIENT mirrors astra.settings.state"
    )
    assert SETTINGS_STATE_TYPE not in set(CLIENT_TO_SERVER), "T-0213 A2r-1: server -> client only"
    assert SETTINGS_STATE_TYPE not in string_set(src, "CLIENT_TYPES", PROTOCOL_JS), (
        "T-0213 A2r-1: the page never sends the state"
    )
    table = model_table(src, SETTINGS_STATE_TYPE)
    declared = set(js_field_names(table))
    fields = set(model.model_fields) - {"type"}
    assert fields <= declared, (
        f"T-0213 A2r-1: the page's astra.settings.state table lacks {sorted(fields - declared)}; a field the "
        "page does not declare turns every live state into a malformed one"
    )
    for name in SETTINGS_STATE_SCALARS:
        assert name in model.model_fields, (
            f"T-0213 A2r-1: AstraSettingsState.{name} does not exist yet"
        )
        entry = re.search(rf"\b{name}\s*:\s*[a-z]+\(([^()]*)\)", table)
        assert entry is not None, f"T-0213 A2r-1: no scalar entry for {name} in the page's table"
        required = model.model_fields[name].is_required()
        assert ("required" in entry.group(1)) == required, (
            f"T-0213 A2r-1: {name} is {'' if required else 'not '}required in the contract and the page says "
            f"{entry.group(1).strip()!r}"
        )


# --- T-0213 A2a-1: the settings set ----------------------------------------------------------------
#
# ``astra.settings.set {rev: int, changes: dict, stop_turn: bool = False}`` is client -> server only. The Python side
# lists it in ``CLIENT_TO_SERVER``; the page's side of that list is its ``MODELS`` table (the page's own egress is
# decoded for parity, as ``astra.mic.mute`` and ``astra.turn.stop`` are), while ``SERVER_TO_CLIENT`` and T-0008's
# ``CLIENT_TYPES`` stay without it (group C's ``hud/src/settings/gate.js`` is what may send it). The fields and their
# ``required`` flags are typed from A2a-1 and compared with the contract's ``model_fields``.

SETTINGS_SET_TYPE = "astra.settings.set"
#: A2a-1, typed from the criterion: field -> required
SETTINGS_SET_FIELDS = {"rev": True, "changes": True, "stop_turn": False}


def test_the_settings_set_is_client_to_server_on_both_sides_with_its_fields():
    """T-0213 A2a-1: ``AstraSettingsSet`` is in the contract's ``CLIENT_TO_SERVER`` and not ``SERVER_TO_CLIENT``,
    with exactly ``rev``, ``changes`` (required) and ``stop_turn`` (optional); the page's ``MODELS`` table declares the
    same three with the same ``required``, and the page's ``SERVER_TO_CLIENT`` and ``CLIENT_TYPES`` do not list it."""
    from local_astra.contracts import protocol

    model = getattr(protocol, "AstraSettingsSet", None)
    assert model is not None, (
        "T-0213 A2a-1: contracts/protocol.py AstraSettingsSet (astra.settings.set) does not exist yet"
    )
    assert SETTINGS_SET_TYPE in set(CLIENT_TO_SERVER), sorted(CLIENT_TO_SERVER)
    assert SETTINGS_SET_TYPE not in set(SERVER_TO_CLIENT), "T-0213 A2a-1: client -> server only"
    fields = {n: f.is_required() for n, f in model.model_fields.items() if n != "type"}
    assert fields == SETTINGS_SET_FIELDS, (
        f"T-0213 A2a-1: the contract's fields are {SETTINGS_SET_FIELDS}; got {fields}"
    )
    src = read_page_module(PROTOCOL_JS, "T-0213 A2a-1")
    assert SETTINGS_SET_TYPE not in string_set(src, "SERVER_TO_CLIENT", PROTOCOL_JS), (
        "T-0213 A2a-1: the page never dispatches its own set"
    )
    assert SETTINGS_SET_TYPE not in string_set(src, "CLIENT_TYPES", PROTOCOL_JS), (
        "T-0213 A2a-1: T-0008's CLIENT_TYPES is not widened; the settings panel's gate sends the set"
    )
    table = model_table(src, SETTINGS_SET_TYPE)
    declared = js_field_names(table)
    assert sorted(declared) == sorted(SETTINGS_SET_FIELDS), (
        f"T-0213 A2a-1: the page's astra.settings.set table declares {sorted(SETTINGS_SET_FIELDS)} once each; got "
        f"{declared}"
    )
    for name, required in SETTINGS_SET_FIELDS.items():
        entry = re.search(rf"\b{name}\s*:\s*[a-zA-Z]+\(([^()]*)\)", table)
        assert entry is not None, (
            f"T-0213 A2a-1: no entry for {name} in the page's table:\n{table[:400]}"
        )
        assert ("required" in entry.group(1)) == required, (
            f"T-0213 A2a-1: {name} is {'' if required else 'not '}required in the contract and the page says "
            f"{entry.group(1).strip()!r}"
        )


# --- T-0227: the page's capture intervals mirror config (docs/tasks/T-0227.md B-1; group B's frames.js) --------


def int_array(src: str, name: str, path: Path) -> tuple[int, ...]:
    """Read ``name = [n, n, ...]`` (optionally ``Object.freeze([...])``) out of a JS module as integers."""
    m = re.search(rf"\b{name}\s*=\s*(?:Object\.freeze\(\s*)?\[([^\]]*)\]", src)
    assert m is not None, (
        f"T-0227 B-1: {path.name} must define {name} as an array literal of integers so this parity test can "
        f"compare it with config.{name}; not found in {path.relative_to(REPO)}"
    )
    items = [t.strip().replace("_", "") for t in m.group(1).split(",") if t.strip()]
    assert all(re.fullmatch(r"\d+", t) for t in items), (
        f"{name}: integer literals only; got {items}"
    )
    return tuple(int(t) for t in items)


def test_the_pages_frame_intervals_equal_config():
    """T-0227 B-1 (parity row, paired with ``tests/hud/frame_interval.test.mjs``): ``FRAME_INTERVALS_MS`` in
    ``hud/src/frames.js`` equals ``config.FRAME_INTERVALS_MS``, element for element."""
    from local_astra import config

    want = getattr(config, "FRAME_INTERVALS_MS", None)
    assert want is not None, "T-0227 A-1: config.FRAME_INTERVALS_MS does not exist yet"
    src = read_page_module(FRAMES_JS, "criterion 5")
    page = int_array(src, "FRAME_INTERVALS_MS", FRAMES_JS)
    assert page == tuple(want)
    assert 1000 / number(src, "MAX_FPS", FRAMES_JS) in page, (
        "the page's default gap is one of its intervals"
    )


def test_the_pages_default_gap_is_the_config_default_interval():
    """T-0227 A-1/B-1: the page's default gap, ``1000 / MAX_FPS`` from ``frames.js`` (the existing MAX_FPS row pins
    MAX_FPS itself), equals ``config.DEFAULT_FRAME_INTERVAL_MS`` (group A only; membership in the page's
    ``FRAME_INTERVALS_MS`` is checked by the group B row above)."""
    from local_astra import config

    default = getattr(config, "DEFAULT_FRAME_INTERVAL_MS", None)
    assert default is not None, "T-0227 A-1: config.DEFAULT_FRAME_INTERVAL_MS does not exist yet"
    src = read_page_module(FRAMES_JS, "criterion 5")
    assert 1000 / number(src, "MAX_FPS", FRAMES_JS) == default
