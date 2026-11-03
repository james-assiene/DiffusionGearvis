"""T-0213 A1-4: the user-instructions block of the system message and its validator.

The interfaces are the task's: ``prompt_builder.USER_INSTRUCTIONS_HEADER`` (a literal), ``MAX_USER_INSTRUCTIONS =
1000``, ``validate_instructions(text) -> str | None`` and ``PromptBuilder(..., user_instructions: str | None =
None)``. Where the task is silent, the choice is fixed here and listed in the test-author's report:

- ``validate_instructions`` returns ``None`` when the text is accepted and the refusal reason (a non-empty str)
  otherwise (ruling R-3).
- Control characters (ruling R-2): tab (U+0009) and ``\\n`` are allowed; every other code point of category
  ``Cc``, ``Cf`` or ``Cs`` is refused, wherever it sits (inside the text or around it: the task's rule has no
  "surrounding whitespace" exemption, that exemption belongs to the length bound only). ``Cf`` includes the bidi
  controls (U+061C, U+200E/F, U+202A-E, U+2066-9) and also the zero-width joiner U+200D, so an emoji ZWJ sequence
  (family, profession emoji) is refused: R-2's consequence, stated in its own row.
- The special-token texts (ruling R-1): the task's list, plus every literal of the pinned tokenizer's
  ``toolcall.special_tokens.SPECIAL_TOKENS``, the bar-less ``<pad>``, ``<eos>``, ``<bos>``, ``<unk>`` and
  ``<mask>`` included. Q4's template pin is a sha only in ``services/versions.lock``; the template text is not in
  the repo.
- With the block, the system text is BASE's text, then ``"\\n"`` (the existing join), then
  ``USER_INSTRUCTIONS_HEADER + "\\n" + text``: "appends, as the last part" of the ``"\\n".join(parts)`` message.

BASE shas: sha256 of ``PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=T, tool_guidance=cli._tool_guidance(v, web,
wait)).system_message().text()`` at BASE 82ee4503, computed once and written out as literals. The rows that pin
them without the new keyword hold at BASE; the rest were written ahead of A1, activated at A1 (implementation
d7a5e87d), and fail at BASE on a named ``AssertionError`` ("... does not exist yet").

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import hashlib
import inspect
import unicodedata
from typing import Any

import pytest

from local_astra import cli
from local_astra.conductor import prompt_builder as pb
from local_astra.conductor.prompt_builder import PromptBuilder
from local_astra.conductor.session import DEFAULT_SYSTEM_PROMPT
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS

HEADER = "The user's own instructions follow. Apply them unless they conflict with anything above."
TOOLS: tuple[dict[str, Any], ...] = ({"type": "function", "function": {"name": "x"}},)

#: (mode, guidance version, web, wait tool) -> sha256 of the system text at BASE 82ee4503
BASE_SHA: dict[tuple[str, str | None, bool, bool], str] = {
    ("off", None, False, False): "2be8295fd01a75bb473de2be57f5367754b25a598b41ce4aeae78e6b9a5b0493",
    (
        "dry_run",
        None,
        False,
        False,
    ): "869f8f96cc4eaccb72a060d359a1456f26225a047b43233bc65256303bd3223e",
    ("live", "v1", True, True): "428142f149ad844efaf871c97b61295c13715f7ff570e77fdc46f514e34f07b7",
    ("live", "v1", True, False): "7186fb4a3bc4be41e40f1ce643a0222d82306ea68751d48bbb93b51e73c32ead",
    ("live", "v1", False, True): "428142f149ad844efaf871c97b61295c13715f7ff570e77fdc46f514e34f07b7",
    (
        "live",
        "v1",
        False,
        False,
    ): "7186fb4a3bc4be41e40f1ce643a0222d82306ea68751d48bbb93b51e73c32ead",
    ("live", "v2", True, True): "ee34418aaae2b16b2924d45895ce16397b1e18282b3bf680cba31634b5045dac",
    ("live", "v2", True, False): "ec173a65a797a1e093fe235b5c7d6956184e94cb3dcff66833e6aa7602327049",
    ("live", "v2", False, True): "ee34418aaae2b16b2924d45895ce16397b1e18282b3bf680cba31634b5045dac",
    (
        "live",
        "v2",
        False,
        False,
    ): "ec173a65a797a1e093fe235b5c7d6956184e94cb3dcff66833e6aa7602327049",
    ("live", "v3", True, True): "235d37af8f067c927f6030c172d060bb3389e7bde61d05ff9f49d475b73a47fb",
    ("live", "v3", True, False): "ee4792639cd076e6125486062825522882828a23483c9258d16cd61ab581d84a",
    ("live", "v3", False, True): "ee34418aaae2b16b2924d45895ce16397b1e18282b3bf680cba31634b5045dac",
    (
        "live",
        "v3",
        False,
        False,
    ): "ec173a65a797a1e093fe235b5c7d6956184e94cb3dcff66833e6aa7602327049",
    ("live", "v4", True, True): "a65934fd5309cfbb4832b0b742e8203e42448f88ba0f6422a8ee70c05a2005fd",
    ("live", "v4", True, False): "7639be45f7fa33559c81f206385a956c0b95da0841e949d3b89a3952ec1c9f80",
    ("live", "v4", False, True): "ee34418aaae2b16b2924d45895ce16397b1e18282b3bf680cba31634b5045dac",
    (
        "live",
        "v4",
        False,
        False,
    ): "ec173a65a797a1e093fe235b5c7d6956184e94cb3dcff66833e6aa7602327049",
}
CASES = sorted(BASE_SHA, key=repr)


def _builder(case: tuple[str, str | None, bool, bool], **kw: Any) -> PromptBuilder:
    mode, version, web, wait = case
    if mode == "off":
        return PromptBuilder(DEFAULT_SYSTEM_PROMPT, **kw)
    if mode == "dry_run":
        return PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=TOOLS, **kw)
    return PromptBuilder(
        DEFAULT_SYSTEM_PROMPT,
        tools=TOOLS,
        tool_guidance=cli._tool_guidance(version, web, wait),
        **kw,
    )


def _sha(b: PromptBuilder) -> str:
    return hashlib.sha256(b.system_message().text().encode("utf-8")).hexdigest()


def _with_instructions(case: tuple[str, str | None, bool, bool], text: str | None) -> PromptBuilder:
    params = inspect.signature(PromptBuilder.__init__).parameters
    assert "user_instructions" in params, (
        f"T-0213 A1-4: PromptBuilder(..., user_instructions: str | None = None) does not exist yet; it takes {sorted(params)}"
    )
    return _builder(case, user_instructions=text)


def _header() -> str:
    value = getattr(pb, "USER_INSTRUCTIONS_HEADER", None)
    assert isinstance(value, str), (
        "T-0213 A1-4: prompt_builder.USER_INSTRUCTIONS_HEADER does not exist yet"
    )
    return value


def _check(text: str) -> str | None:
    fn = getattr(pb, "validate_instructions", None)
    assert callable(fn), "T-0213 A1-4: prompt_builder.validate_instructions does not exist yet"
    out = fn(text)
    assert out is None or (isinstance(out, str) and out.strip()), (
        f"validate_instructions returns None or a reason: {out!r}"
    )
    return out


def _accepts(text: str) -> None:
    reason = _check(text)
    assert reason is None, f"{text[:40]!r} (len {len(text)}) must be accepted; refused: {reason}"


def _refuses(text: str) -> str:
    reason = _check(text)
    assert reason is not None, f"{text[:60]!r} (len {len(text)}) must be refused"
    return reason


# --- BASE bytes -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=repr)
def test_base_system_message_shas(case: tuple[str, str | None, bool, bool]) -> None:
    """Holds at BASE: the literal shas are the system message as BASE builds it, for every mode x guidance."""
    assert _sha(_builder(case)) == BASE_SHA[case]


@pytest.mark.parametrize("text", [None, ""])
@pytest.mark.parametrize("case", CASES, ids=repr)
def test_empty_block_is_base_bytes(
    case: tuple[str, str | None, bool, bool], text: str | None
) -> None:
    """A1-4 negative: with ``None`` or ``""`` the system message is BASE's, byte for byte."""
    assert _sha(_with_instructions(case, text)) == BASE_SHA[case]


# --- the block ---------------------------------------------------------------------------------------------------


def test_header_and_bound_literals() -> None:
    assert _header() == HEADER
    assert getattr(pb, "MAX_USER_INSTRUCTIONS", None) == 1000, (
        "T-0213 A1-4: MAX_USER_INSTRUCTIONS = 1000"
    )


@pytest.mark.parametrize(
    "case",
    [
        ("off", None, False, False),
        ("dry_run", None, False, False),
        ("live", "v3", True, True),
        ("live", "v2", False, False),
    ],
    ids=repr,
)
def test_block_is_last(case: tuple[str, str | None, bool, bool]) -> None:
    text = "Answer in French.\nKeep answers under three sentences."
    base = _builder(case).system_message().text()
    got = _with_instructions(case, text).system_message()
    assert got.role == "system" and len(got.content) == 1, "still one text part"
    full = got.text()
    assert full.startswith(base), "the block extends BASE's text, it never edits it"
    assert full[len(base) :] == "\n" + HEADER + "\n" + text, (
        f"the block is last: {full[len(base) :]!r}"
    )


def test_block_reaches_only_the_system_message_and_keeps_thinking() -> None:
    """``<|think|>`` stays the first system token: the flag rides on the template kwarg, never on the text."""
    b = _with_instructions(("live", "v3", True, True), "Answer in French.")
    assert b.thinking is True
    assert "<|think|>" not in b.system_message().text()
    off = _with_instructions(("live", "v3", True, True), "Answer in French.")
    off_thinking = PromptBuilder(DEFAULT_SYSTEM_PROMPT, thinking=False, tools=TOOLS, user_instructions="Answer in French.",
                                 tool_guidance=cli._tool_guidance("v3", True, True))  # fmt: skip
    assert off_thinking.system_message() == off.system_message(), (
        "thinking on/off share the system bytes"
    )


def test_builders_are_independent() -> None:
    """Two builders with different instructions do not share state, and a BASE builder after them is BASE's."""
    case = ("live", "v3", True, True)
    a = _with_instructions(case, "Answer in French.").system_message().text()
    b = _with_instructions(case, "Answer in German.").system_message().text()
    assert a != b and a.endswith("French.") and b.endswith("German.")
    assert _sha(_with_instructions(case, None)) == BASE_SHA[case]


# --- the bound ---------------------------------------------------------------------------------------------------


def test_bound_counts_code_points_with_astral_characters() -> None:
    _accepts("a" * 1000)
    _refuses("a" * 1001)
    astral = "\U0001f600"  # 1 code point, 2 UTF-16 units, 4 UTF-8 bytes
    _accepts(astral * 1000)
    _refuses(astral * 1001)
    _refuses(astral * 999 + "ab")
    _accepts(astral * 500 + "a" * 500)
    # combining marks are code points of their own: 500 graphemes, 1000 code points
    decomposed = "é"
    assert unicodedata.normalize("NFC", decomposed) != decomposed
    _accepts(decomposed * 500)
    _refuses(decomposed * 500 + "e")


def test_bound_is_after_stripping_surrounding_whitespace() -> None:
    _accepts("   " + "a" * 1000 + "\n\n  ")
    _refuses("   " + "a" * 1001 + "  ")
    _refuses("a" * 500 + " " * 2 + "a" * 499)  # inner whitespace counts: 1001


def test_ordinary_text_is_accepted() -> None:
    """The guard's too-wide direction: text near the banned shapes, and newlines, pass."""
    for text in (
        "Answer in French.",
        "Line one\nLine two\n\nLine four",
        "x <= y and a | b > c",
        "Use <b>bold</b> sparingly; return early; turn left at the |sign|",
        "Think step by step; the turnout was high.",
        "Gearvis, prefer metric units — not imperial.",
        "",
    ):
        _accepts(text)


# --- adversarial: special tokens, split tokens, control and bidi characters -----------------------------------------

#: code points: C0 controls, CR, DEL, NEL (C1), and the bidi controls ALM, LRM, RLM, LRE, RLE, LRO, RLO,
#: LRI, RLI, FSI, PDI (tab is allowed, R-2: its own row below)
CONTROL_AND_BIDI = (
    0x00, 0x07, 0x1B, 0x0D, 0x7F, 0x85,
    0x061C, 0x200E, 0x200F, 0x202A, 0x202B, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069,
)  # fmt: skip
BIDI_CLASSES = {"AL", "L", "R", "LRE", "RLE", "LRO", "RLO", "LRI", "RLI", "FSI", "PDI"}
#: other format (Cf) characters R-2 refuses, and a lone surrogate (Cs): ZWSP-adjacent joiners ZWNJ, ZWJ, the word
#: joiner, the BOM / ZWNBSP and the soft hyphen
OTHER_CF_CS = (0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0xD800)

TASK_TOKENS = ("<|", "|>", "<turn", "turn|>", "<|think|>", "<|tool_call>", "<|image|>")
#: R-1: every literal of the pinned tokenizer, with or without bars
PIN_TOKENS = tuple(SPECIAL_TOKENS)
#: R-1: the bar-less specials, written out from the ruling (an oracle independent of the SPECIAL_TOKENS table)
BARLESS = ("<eos>", "<bos>", "<pad>", "<unk>", "<mask>")


@pytest.mark.parametrize("token", sorted(set(TASK_TOKENS) | set(PIN_TOKENS)))
def test_special_token_text_is_refused(token: str) -> None:
    _refuses(token)
    _refuses(f"Please answer in French. {token} Thank you.")
    _refuses(f"Please answer\n{token}\nin French.")


def test_barless_premise() -> None:
    """The ruling's five literals are the pinned tokenizer's ids 0-4, so the parametrised row above covers them."""
    assert set(BARLESS) <= set(SPECIAL_TOKENS)


@pytest.mark.parametrize("token", BARLESS)
def test_barless_special_tokens_are_refused(token: str) -> None:
    """R-1: ``<eos>``, ``<bos>``, ``<pad>``, ``<unk>`` and ``<mask>`` are refused, alone and inside text."""
    _refuses(token)
    _refuses(f"Stop at {token} when done.")
    _refuses(f"first line\n{token}")


def test_barless_guard_is_not_too_wide() -> None:
    """The too-wide direction of R-1: ordinary angle-bracket text near the five literals passes."""
    for text in (
        "Use <b>bold</b> and <em>emphasis</em>.",
        "pad the margins; mask the noise; eos is an abbreviation",
        "the <padding> element and <unknown> tags",
    ):
        _accepts(text)


@pytest.mark.parametrize(
    "text",
    [
        "<|" + "\nthink|>",  # the task's split across lines
        "be brief <|\nthink|> really",
        "<turn" + "|>",
        "<tu" + "rn|>",  # 'turn|>' formed across the join
        "ok |>",
        "<|",
        "end with <turn",
    ],
)
def test_split_token_text_is_refused(text: str) -> None:
    _refuses(text)


# built with chr() so no raw control or bidi character sits in this source file (Trojan Source)
_CHAR_IDS = {"ids": lambda c: f"U+{ord(c):04X}"}


@pytest.mark.parametrize("char", [chr(c) for c in CONTROL_AND_BIDI], **_CHAR_IDS)
def test_control_and_bidi_characters_are_refused(char: str) -> None:
    _refuses(f"Answer in French.{char}Always.")


@pytest.mark.parametrize("char", [chr(c) for c in CONTROL_AND_BIDI], **_CHAR_IDS)
def test_surrounding_control_characters_are_refused(char: str) -> None:
    """R-2 has no position exemption: a control character before or after the text, or alone, is refused too.
    (``\\r`` and NEL are also ``str.isspace``; the whitespace strip is the length bound's rule, not a licence.)"""
    _refuses(char)
    _refuses(f"{char}Answer in French.")
    _refuses(f"Answer in French.{char}")


@pytest.mark.parametrize("char", [chr(c) for c in OTHER_CF_CS], **_CHAR_IDS)
def test_other_format_and_surrogate_characters_are_refused(char: str) -> None:
    """R-2: every other ``Cf`` and ``Cs`` code point is refused, not only the bidi ones."""
    _refuses(f"Answer in French.{char}Always.")


def test_zwj_emoji_sequence_is_refused() -> None:
    """R-2's stated consequence: the zero-width joiner U+200D is ``Cf``, so an emoji ZWJ sequence (man, ZWJ,
    laptop: "technologist") is refused, while the same emoji without the joiner passes."""
    man, zwj, laptop = chr(0x1F468), chr(0x200D), chr(0x1F4BB)
    assert unicodedata.category(zwj) == "Cf"
    _refuses(f"Sign off with {man}{zwj}{laptop}.")
    _accepts(f"Sign off with {man}{laptop}.")


def test_control_and_bidi_premise() -> None:
    """Holds at BASE: the lists are what their comments claim (C0, DEL, C1 are ``Cc``; the bidi ones ``Cf``; the
    others ``Cf`` or ``Cs``), and tab is not in the refused list."""
    cats = {c: unicodedata.category(chr(c)) for c in CONTROL_AND_BIDI}
    assert {c for c, k in cats.items() if k == "Cc"} == {0x00, 0x07, 0x1B, 0x0D, 0x7F, 0x85}
    assert all(
        unicodedata.bidirectional(chr(c)) in BIDI_CLASSES for c, k in cats.items() if k == "Cf"
    )
    assert {unicodedata.category(chr(c)) for c in OTHER_CF_CS} == {"Cf", "Cs"}
    assert 0x09 not in CONTROL_AND_BIDI


def test_newline_and_tab_are_the_control_characters_allowed() -> None:
    """R-2: ``\\n`` and tab (U+0009) are allowed, inside the text and around it."""
    tab = chr(0x09)
    _accepts("a\nb")
    _accepts("\n".join(["line"] * 10))
    _accepts(f"name{tab}value")
    _accepts(f"col1{tab}col2\nx{tab}y")
    _accepts(f"{tab}indented line")


def test_tab_reaches_the_block() -> None:
    """R-2: an allowed tab inside the text reaches the system message as written."""
    tab = chr(0x09)
    text = f"Units:{tab}metric"
    full = _with_instructions(("live", "v3", True, True), text).system_message().text()
    assert full.endswith(HEADER + "\n" + text)


# --- review rows (A1 review at 5718460a): the builder's own seams -------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["say <|think|> now", "x" * 1001, "left" + chr(0x202E) + "right", "stop at <eos>"],
    ids=["special", "too_long", "bidi", "barless"],
)
def test_builder_refuses_text_the_validator_refuses(text: str) -> None:
    """M10: ``PromptBuilder(user_instructions=...)`` is a second door into the system message, so text that
    ``validate_instructions`` refuses raises ``ValueError`` there too (never reaches the prompt)."""
    assert _check(text) is not None, f"premise: {text[:20]!r} is refused by the validator"
    with pytest.raises(ValueError):
        _with_instructions(("live", "v3", True, True), text)


def test_builder_accepts_what_the_validator_accepts() -> None:
    """M10's too-wide direction: accepted text (tab, newlines, the 1000-code-point bound) builds."""
    for text in ("Units:" + chr(0x09) + "metric", "a\nb", "a" * 1000, "  " + "a" * 1000 + "\n"):
        assert _check(text) is None
        _with_instructions(("live", "v3", True, True), text).system_message()


@pytest.mark.parametrize(
    "case", [("off", None, False, False), ("live", "v3", True, True)], ids=repr
)
def test_surrounding_whitespace_is_stripped_in_the_block(
    case: tuple[str, str | None, bool, bool],
) -> None:
    """M11: the bound is counted after stripping surrounding whitespace, and the block carries the stripped text:
    the system message ends with the header, one newline and the text, no leading or trailing blank."""
    base = _builder(case).system_message().text()
    raw = "\n  " + chr(0x09) + "Answer in French.\nBe brief. " + chr(0x09) + "\n\n "
    full = _with_instructions(case, raw).system_message().text()
    assert full[len(base) :] == "\n" + HEADER + "\n" + "Answer in French.\nBe brief.", (
        f"the block: {full[len(base) :]!r}"
    )
