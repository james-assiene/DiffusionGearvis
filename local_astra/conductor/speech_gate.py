"""Speech gate: committed *answer* text → sentences for TTS (docs/03 §3). Task T-0004.

Contract (tests pin it):

- input is only ``Commit(segment="answer")`` text; ``thought``/``ui`` segments and any commit with
  ``thought_open=True`` are rejected *before* the text enters the buffer (``feed`` raises) so a
  rejected commit can never surface on a later ``close()``;
- splits on sentence boundaries with abbreviations/decimals/URLs handled (they stay verbatim);
- holds back an unfinished clause until the next commit or ``close()`` (multi-canvas answers);
- ``normalise_for_tts`` rewrites only what Pocket TTS cannot read: file paths, code identifiers
  (camelCase / snake_case → spaced words), fenced code — closed or left open at the end of the
  canvas, both → "a code block" (T-0022 R1) — and stray special tokens / text after EOS.
  Numbers, dates, currency and URLs are left to the TTS engine — whether that is good enough is
  measured in X4, not assumed here;
- every emitted ``Sentence`` carries ``source`` (the exact committed substring, what the §4
  invariant checks) and ``text`` (the normalised rendering handed to TTS);
- property: the concatenation of emitted ``text`` equals ``normalise_for_tts`` of the concatenated
  committed answer text (nothing lost, nothing invented) — *except* for sources the voice guard
  refuses, which are counted and retrievable (T-0052 ruling A5 amends docs/03 §"Speech gate");
- the voice guard (T-0052 criteria 8–11): a sentence whose committed source is structurally A2UI
  interface data is refused rather than voiced. See ``_is_interface_data``. A refusal never breaks
  byte conservation: the refused sentence is still returned with its ``source`` intact and
  ``text=""`` (ruling A20), so only the *voicing* is lost;
- the repeat stop (T-0224 group A): once an answer sentence re-says one already voiced this turn
  (``_repeat_key`` equal, or contained at word boundaries, at least ``REPEAT_MIN_CHARS`` long), it
  and every later voiceable sentence of the turn are refused the same way, reason ``"repeat"``;
- the instruction stop (T-0232): an answer sentence that voices Gearvis's own system instructions
  (``INSTRUCTION_SENTENCES``, derived from ``prompt_builder`` at import) is refused, reason
  ``"instructions"``, without latching the turn; see ``_is_instruction``;
- maths markup (T-0224 group B): ``speak_maths`` turns the LaTeX inside ``$…$``/``$$…$$`` and
  inline backtick spans into words inside ``normalise_for_tts`` (prose outside a span is main's,
  ruling R-6), so only the spoken ``text`` changes.

Rulings implemented here (docs/tasks/T-0004.md amendments):

- **A1, conservation is byte-exact.** ``"".join(source) + pending`` rebuilds every byte fed so far,
  at every step — the turn machine (T-0003 criterion 10) raises ``ValueError`` otherwise. The
  whitespace following a terminator rides on the *preceding* sentence's ``source``; ``text`` is
  that source normalised (which strips it), so ``pending`` always starts at a word.
- **A2/A3, special tokens.** An EOS token (``<end_of_turn>``, ``<eos>``, ``<|endoftext|>``) means
  the model stopped: everything from it on is trimmed out of ``text``. ``<|think|>`` trims too —
  it switches the model back to the thought channel, and thinking is never voiced (CLAUDE.md).
  Any other ``<|…|>`` token is a tokenizer artifact: it is dropped in place and the surrounding
  text joined. T-0088a criterion 3 extends both to every Gemma 4 special literal
  (``toolcall.special_tokens.SPECIAL_TOKENS``): ``<turn|>``, ``<|tool_response>``, ``<|tool_call>``
  and ``<|channel>`` trim too, every other literal is deleted with nothing inserted (D2), repeated
  until none is left, and no ``text`` ever contains one. The trimmed/dropped bytes still travel inside a ``source`` (conservation) with a
  trimmed or empty ``text`` (so TTS never voices them), and after ``close()`` nothing is held.
- **A4.** Segmentation and normalisation go exactly as far as the task's fixtures require; the
  four rules below are the whole rule, not a sample: the abbreviation list (``_ABBREVIATIONS``,
  matched case-insensitively — A10), the path/identifier shapes (``_PATH_RE``/``_IDENT_RE``), the
  balance hold and the closer attachment.
- **A7, the balance hold and the closers.** A terminator inside an open parenthesis/bracket or an
  open ``` ``` ``` fence is not a sentence end: ``_is_balanced`` holds the clause until it closes
  (or until ``close()``). Double quotes are *not* part of that rule — see ``_is_balanced``.
  Conversely a closer in ``_CLOSERS`` (``"``, ``'``, ``)``, ``]``, ``}``) that follows a terminator
  stays attached to the sentence it ends rather than starting the next one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from local_astra.conductor import prompt_builder
from local_astra.contracts.llm import Commit
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS, drop_special_literals


class GateInputError(ValueError):
    """Raised when non-answer or unclosed-thought text is fed to the gate."""


@dataclass(frozen=True)
class Sentence:
    source: str  # exact substring of the committed answer
    text: str  # TTS-normalised rendering


# --- segmentation rules ---------------------------------------------------------------------

#: Sentence-final punctuation. A run of them ("?!", "!!") ends one sentence, an ellipsis does not.
_TERMINATORS = ".!?"

#: Quotes/brackets that may close *after* the terminator and still belong to the sentence.
_CLOSERS = "\"')]}"

#: Tokens that never end a sentence even when a space follows (A4: this list is the whole rule).
_ABBREVIATIONS = frozenset(
    {
        "e.g.",
        "i.e.",
        "cf.",
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "vs.",
        "approx.",
        "fig.",
    }
)

_OPENERS = "([{"
_CLOSING_BRACKETS = ")]}"

# --- normalisation rules --------------------------------------------------------------------

#: Everything from one of these on is trimmed out of the spoken text (A2): the model has either
#: stopped (EOS), gone back to thinking (``<|think|>``, ``<|channel>``), or started something that is
#: not speech (``<|tool_call>``, ``<|tool_response>``), and none of it is answer text. The Gemma 4
#: ones are T-0088a criterion 3 "Trimmed"; ``<end_of_turn>``/``<|endoftext|>`` are main's.
_TRIM_TOKENS: tuple[str, ...] = (
    "<end_of_turn>",
    "<eos>",
    "<|endoftext|>",
    "<|think|>",
    "<turn|>",
    "<|tool_response>",
    "<|tool_call>",
    "<|channel>",
)

#: Every other Gemma 4 special literal (T-0088a criterion 3 "Dropped"): deleted in place, with
#: nothing inserted (ruling D2). Derived from the frozen table so that no literal is left out.
_DROP_TOKENS: tuple[str, ...] = tuple(t for t in SPECIAL_TOKENS if t not in _TRIM_TOKENS)

#: Any other ``<|…|>`` token: a tokenizer artifact, dropped in place.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")

#: A fenced code block. The closing ``` is optional: a canvas cut off by ``llm.done`` or
#: ``max_tokens`` mid-fence is a fence running to the end of the input and gets the same
#: marker (T-0022 R1). Non-greedy, so prose between and after two closed fences survives.
_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)

#: A file path: starts at ``~``/``.``/``..`` or at a ``/`` that is not inside a URL or another path
#: (the lookbehind is what keeps ``https://example.com/a.b`` verbatim). A trailing ``.`` is left
#: outside the match so sentence punctuation survives.
_PATH_RE = re.compile(r"(?<![\w:/~.])(?:~|\.{1,2})?(?:/[\w+-]+(?:\.[\w+-]+)*)+/?")

#: A code identifier: a call ``foo()``, snake_case, or camelCase.
_IDENT_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*\(\)"
    r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"
    r"|\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b"
)

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


# --- the voice guard (T-0052 group B, criteria 8–11 and 15) ---------------------------------
#
# The *second* line of defence (T-0052 D5, ruling A2). The fence carry in ``llm_client`` is the
# real fix; this exists so that a future regression of that classifier becomes a missing surface
# and a counted refusal instead of Pocket TTS reading 807 characters of A2UI JSON aloud, as it did
# on 2026-09-16. It is the one guarantee that does not depend on the classifier being right: the
# gate is the single funnel every committed answer string passes through on its way to TTS.
#
# **It must be narrow.** A false positive silences a real answer, which is worse than the bug it
# prevents and invisible to the human — nobody reports a sentence the assistant never said. Both
# rules below therefore require a literal ``{``, so prose about components, children and surfaces,
# and quote-and-comma dense dialogue, stay spoken byte-exact (criterion 15).
#
# --- the measurements both constants are set from (T-0052 ruling A37) -------------------------
#
# The one place these numbers live. Three rulings written on the evening of 2026-09-16 quoted
# three different roundings of this same table, from which a later reader would conclude the
# measurements disagree; the constants below therefore state their *rule* and point here rather
# than restating a value. Structural density is ``{}[]":,`` characters over length — the same
# arithmetic the density rule does. Regime: pure CPU, n = 1 per row, no seed, recomputed against
# this tree on 2026-09-16.
#
# Real A2UI, from ``tests/fixtures/llm/flight_canvases.json`` (the human's own session):
#
#   ``ui_piece_canvas1``      409 chars  0.293   the smaller of the two real surfaces
#   ``json_tail_canvas2``     750 chars  0.365   canvas 2's JSON tail
#   ``tts_utterance_source``  810 chars  0.338   what Pocket TTS read aloud, session log seq 2223
#
# so the interface band is **0.293–0.365 over 409–810 characters**. (Ruling A21 wrote its top as
# 0.373; no payload in the tree measures that, and ``json_tail_canvas2`` computes 0.365.)
#
# Ordinary prose the guard must keep speaking, from ``tests/unit/test_speech_gate_voice_guard.py``:
#
#   JSON-quoting answers, en/fr/ja/ko  52–65 chars  0.262–0.328  *inside* the interface band
#   reported dialogue, one sentence       69 chars  0.217        densest brace-free prose measured
#   the long dense-dialogue row          333 chars  0.225        its length follows the floor
#
# The two populations **overlap in density and separate only in length**, and that is what sets
# each constant below: the fraction sits above ordinary dialogue, the floor above ordinary answer
# length. Neither alone would be narrow enough.

#: The A2UI keys criterion 9 pins, quoted as they appear on the wire: the *word* "component" in
#: prose must not match, only the JSON member ``"component"``. A constant a test reads rather than
#: a literal inside a regex, so widening the guard is a deliberate, reviewable act.
A2UI_KEYS: frozenset[str] = frozenset(
    {
        '"component"',
        '"surfaceId"',
        '"createSurface"',
        '"updateComponents"',
        '"catalogId"',
        '"children"',
        '"version":"v0.9.1"',
    }
)

#: The characters the density rule counts as structural.
STRUCTURAL_CHARS = '{}[]":,'

#: Structural density above which a source that *also* contains ``{`` is interface data (T-0052
#: ruling A9). 0.25 sits in the gap the table above measures: above every ordinary-prose row,
#: including the long dense-dialogue one, and below every real A2UI payload.
STRUCTURAL_FRACTION = 0.25

#: The density rule ignores short sources, and the floor is what makes the rule narrow (T-0052
#: ruling A21): restricted to its own domain — sources containing ``{`` — density does *not*
#: separate interface data from prose, because the JSON-quoting rows of the table above sit inside
#: the interface band. Only length separates them, so the floor, not the fraction, is what keeps an
#: ordinary answer speakable. Ruling A32 makes the band ``65 < STRUCTURAL_MIN_CHARS <= 409`` the
#: contract — above the longest ordinary JSON-quoting answer, at or below the smaller real
#: surface — and leaves the value to this module.
#:
#: 200 is chosen inside it with margin on both sides: about three times the longest JSON-quoting
#: answer, and about half the smaller of the two real surfaces, so both of them stay within the
#: density rule's reach rather than being exempted by a floor set at their own size.
#:
#: Residual, stated rather than hidden: a *single sentence* longer than 200 characters that quotes
#: JSON dense enough to pass :data:`STRUCTURAL_FRACTION` is still a false positive and is silenced.
#: No length separates that case, and a false positive costs one sentence (ruling A20 conserves the
#: rest of the turn) and is counted on the gate.
STRUCTURAL_MIN_CHARS = 200

#: How many refused sources one gate keeps — criterion 10 asks for a *bounded* list. Past the
#: bound refusals are still counted; only the transcript stops growing, so a pathological turn
#: cannot grow the gate without limit.
MAX_REFUSED_SOURCES = 8


#: T-0224 criterion 1: a voiced sentence shorter than this (its repeat key, in characters) never
#: triggers the repeat stop, so "Yes." or "OK." said twice is still said twice.
REPEAT_MIN_CHARS = 20

_TRAILING_PUNCT_RE = re.compile(r"[\s!-/:-@\[-`{-~…]+$")


def _repeat_key(text: str) -> str:
    """The form two sentences are compared in for the repeat stop: casefolded, whitespace
    collapsed, trailing punctuation dropped (T-0224 criterion 1)."""
    key = re.sub(r"\s+", " ", text.casefold()).strip()
    return _TRAILING_PUNCT_RE.sub("", key)


def _is_interface_data(source: str) -> bool:
    """True when ``source`` is structurally A2UI interface data and must not be voiced.

    Criterion 9 as amended by rulings A8 and A9. Two rules, each requiring a JSON object opener:

    1. the source contains ``{`` **and**, separately, a quoted member of :data:`A2UI_KEYS`.
       Adjacency is deliberately *not* required: in the recorded payload every ``{`` is followed
       by ``"id"`` and ``"component"``/``"children"`` always follow a comma, so an adjacency rule
       would have been dead code on the very string it was written to refuse (A8);
    2. for a source longer than :data:`STRUCTURAL_MIN_CHARS` that contains ``{``, structural
       characters exceed :data:`STRUCTURAL_FRACTION` of its length. The ``{`` conjunct is not
       decoration: ``tests/unit/test_speech_gate.py``'s conservation property draws text from an
       alphabet containing ``,`` ``:`` ``"`` ``[`` ``]`` but **not** ``{``, so a density rule
       standing on its own would make a settled property test flaky (A9).

    Pure and total: it inspects the source and never raises, prints or logs (criterion 10).
    """
    if "{" not in source:
        return False
    if any(key in source for key in A2UI_KEYS):
        return True
    if len(source) <= STRUCTURAL_MIN_CHARS:
        return False
    structural = sum(1 for ch in source if ch in STRUCTURAL_CHARS)
    return structural / len(source) > STRUCTURAL_FRACTION


def _trim_index(text: str) -> int | None:
    """Index of the earliest trimming token, or ``None``."""
    found = [i for i in (text.find(tok) for tok in _TRIM_TOKENS) if i != -1]
    return min(found) if found else None


def _drop_tokens(text: str) -> str:
    """``text`` with every ``_DROP_TOKENS`` literal deleted, until none is left (a deletion can
    splice a new one: ``<|chan<pad>nel>``)."""
    while any(tok in text for tok in _DROP_TOKENS):
        for tok in _DROP_TOKENS:
            text = text.replace(tok, "")
    return text


def _cut_and_drop(text: str) -> tuple[str, bool]:
    """T-0088a criterion 3: cut at the earliest trim literal, delete the dropped ones, and repeat,
    because a deletion can splice a trim literal (``<|tool<pad>_call>``). Returns the text and
    whether anything was trimmed. On text with no Gemma 4 special literal this is main's cut."""
    trimmed = False
    while True:
        cut = _trim_index(text)
        if cut is not None:
            text = text[:cut]
            trimmed = True
        dropped = _drop_tokens(text)
        if dropped == text:
            return text, trimmed
        text = dropped


def _say_path(match: re.Match[str]) -> str:
    raw = match.group(0)
    parts = [p for p in raw.split("/") if p]
    if not parts:
        return raw
    base = parts[-1]
    spoken = " dot ".join(p for p in base.split(".") if p) or base
    if len(parts) < 2:
        return spoken
    owner = "your" if parts[0] == "~" else "the"
    parent = "home" if parts[-2] == "~" else parts[-2]
    return f"{spoken} in {owner} {parent} folder"


def _say_identifier(match: re.Match[str]) -> str:
    raw = match.group(0).removesuffix("()")
    words = _CAMEL_BOUNDARY_RE.sub(" ", raw.replace("_", " ")).lower()
    return " ".join(words.split())


# --- maths markup (T-0224 group B) ------------------------------------------------------------
#
# The model writes maths as LaTeX/markdown ("$mg \cos \theta$", "$4.9t^2$"); Pocket TTS reads the
# markup character by character. These rules turn it into words in the *spoken* text only:
# ``source`` keeps the model's bytes, and cards and logs never pass through here. Unicode maths
# already in plain prose (θ, ², √) is not touched. The currency rule: a ``$`` is a maths delimiter
# only when it opens a span the way pandoc's ``tex_math_dollars`` does — the opener is followed by
# a non-space, the closer is preceded by a non-space and not followed by a digit — so "costs $5"
# and "$5 and $10" are left alone for the TTS engine to read as money.

#: ``$$…$$`` display maths, then ``$…$`` inline maths (pandoc's rule, see above).
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$(?=\S)([^$\n]*?\S)\$(?!\d)")
#: An inline code span (single backticks; fenced code is already "a code block" by now).
_INLINE_CODE_RE = re.compile(r"`([^`\n]*)`")
#: ``\(`` ``\)`` ``\[`` ``\]`` maths delimiters.
_TEX_DELIMS_RE = re.compile(r"\\[()\[\]]")

#: Commands with a spoken form other than their own name.
_TEX_WORDS: dict[str, str] = {
    "times": "times",
    "cdot": "times",
    "div": "divided by",
    "pm": "plus or minus",
    "approx": "approximately",
    "neq": "is not equal to",
    "ne": "is not equal to",
    "leq": "is less than or equal to",
    "le": "is less than or equal to",
    "geq": "is greater than or equal to",
    "ge": "is greater than or equal to",
    "infty": "infinity",
    "varepsilon": "epsilon",
    "vartheta": "theta",
    "varphi": "phi",
    "varpi": "pi",
    "varrho": "rho",
    "varsigma": "sigma",
}

#: Layout commands that are not spoken at all (their argument, if any, is kept).
_TEX_SILENT = frozenset(
    {
        "left",
        "right",
        "big",
        "Big",
        "bigg",
        "Bigg",
        "quad",
        "qquad",
        "displaystyle",
        "text",
        "textrm",
        "textbf",
        "textit",
        "mathrm",
        "mathbf",
        "mathit",
        "mathsf",
        "mathcal",
        "mathbb",
        "boldsymbol",
        "operatorname",
        "rm",
        "bf",
    }
)

_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
#: A command and its one argument; ``\frac``/``\sqrt`` are left to their own rules above.
_TEX_ARG_RE = re.compile(r"\\(?!(?:[dt]?frac|sqrt)\b)([A-Za-z]+)\s*\{([^{}]*)\}")
_POWER_DEGREES_RE = re.compile(r"\^\s*(?:\{\s*\\circ\s*\}|\\circ)")
_POWER_GROUP_RE = re.compile(r"\^\s*\{([^{}]*)\}")
_POWER_BARE_RE = re.compile(r"\^\s*(-?[A-Za-z0-9]+)")
_SUB_GROUP_RE = re.compile(r"_\{([^{}]*)\}")
_SUB_DIGITS_RE = re.compile(r"_(\d+)")
_TEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
_TEX_SPACING_RE = re.compile(r"\\[,;:! ]|\\\\")
_TEX_ESCAPED_BRACE_RE = re.compile(r"\\[{}]")


def _spaced(match: re.Match[str], words: str) -> str:
    """``words`` in place of ``match``, with a space added on a side where it would otherwise
    touch a letter or digit (``2\\times2`` → "2 times 2", ``\\cos(x)`` → "cos(x)")."""
    if not words:
        return ""
    s, a, b = match.string, match.start(), match.end()
    before = " " if a > 0 and s[a - 1].isalnum() else ""
    after = " " if b < len(s) and s[b].isalnum() else ""
    return before + words + after


def _say_power(exponent: str) -> str:
    exponent = exponent.strip()
    if exponent == "2":
        return " squared"
    if exponent == "3":
        return " cubed"
    return f" to the power {exponent}"


def _say_command(match: re.Match[str]) -> str:
    name = match.group(1)
    if name in _TEX_SILENT:
        return " " if match.start() > 0 and match.string[match.start() - 1].isalnum() else ""
    return _spaced(match, _TEX_WORDS.get(name, name))


def speak_maths(text: str) -> str:
    """LaTeX/markdown maths markup → words for TTS (T-0224 criterion 4). Pure and total.

    ``$$…$$`` and ``$…$`` spans lose their delimiters, inline backtick spans are unwrapped, and
    the content of each is spoken by :func:`_tex_to_words`. Nothing outside a span is touched
    (ruling R-6): plain prose — a brace, an underscore, a caret, a backslash — is spoken exactly
    as on main. Text with neither ``$`` nor a backtick is returned unchanged.
    """
    if "$" not in text and "`" not in text:
        return text
    text = _DISPLAY_MATH_RE.sub(lambda m: f" {_tex_to_words(m[1])} ", text)
    text = _INLINE_MATH_RE.sub(lambda m: _tex_to_words(m[1]), text)
    return _INLINE_CODE_RE.sub(lambda m: _tex_to_words(m[1]), text)


def _tex_to_words(text: str) -> str:
    """The TeX rules of criterion 4 on maths content: ``\\frac{a}{b}`` → "a over b",
    ``\\sqrt{x}`` → "square root of x", powers → "squared"/"cubed"/"to the power n",
    ``_{1}``/``_1`` → " 1", every other ``\\name`` → its spoken word (``_TEX_WORDS``) or its
    name, layout commands → nothing, stray braces removed."""
    text = _TEX_DELIMS_RE.sub(" ", text)
    text = _TEX_SPACING_RE.sub(" ", text)
    text = _TEX_ESCAPED_BRACE_RE.sub("", text)
    text = _POWER_DEGREES_RE.sub(" degrees", text)
    # Innermost groups first, until nothing changes: ``\frac{\sqrt{2}}{2}`` needs two passes.
    while True:
        before = text
        text = _FRAC_RE.sub(lambda m: _spaced(m, f"{m[1].strip()} over {m[2].strip()}"), text)
        text = _SQRT_RE.sub(lambda m: _spaced(m, f"square root of {m[1].strip()}"), text)
        text = _POWER_GROUP_RE.sub(lambda m: _say_power(m[1]), text)
        text = _SUB_GROUP_RE.sub(lambda m: f" {m[1].strip()}", text)
        text = _TEX_ARG_RE.sub(
            lambda m: f" {m[2]} " if m[1] in _TEX_SILENT else _spaced(m, f"{m[1]} {m[2]}"),
            text,
        )
        if text == before:
            break
    text = _POWER_BARE_RE.sub(lambda m: _say_power(m[1]), text)
    text = _SUB_DIGITS_RE.sub(r" \1", text)
    text = _TEX_COMMAND_RE.sub(_say_command, text)
    return text.replace("{", "").replace("}", "")


def normalise_for_tts(text: str) -> str:
    """File paths, code identifiers, fenced code and special tokens → speakable text.

    Deliberately narrow (see module docstring); identity on ordinary prose, numbers, dates and
    URLs. Leading/trailing and repeated whitespace is collapsed: ``text`` is what reaches the TTS,
    not what the model committed (``source`` carries that, byte for byte).
    """
    return _normalise(text)[0]


def _normalise(text: str) -> tuple[str, bool]:
    """``normalise_for_tts`` plus whether a trim literal ended the speakable text."""
    text, trimmed = _cut_and_drop(text)
    text = _FENCE_RE.sub(" a code block. ", text)
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    text = _PATH_RE.sub(_say_path, text)
    text = _IDENT_RE.sub(_say_identifier, text)
    # T-0224: after the path/identifier rewrites (so a path's ``_1`` is not taken for a
    # subscript); a removed brace can splice a ``<|…|>`` artifact, so that rule runs again — but
    # only on text the maths rules changed, so text without maths markup is main's, byte for byte.
    spoken = speak_maths(text)
    if spoken != text:
        text = _SPECIAL_TOKEN_RE.sub(" ", spoken)
    text = re.sub(r"\s+", " ", text).strip()
    if any(tok in text for tok in SPECIAL_TOKENS):
        # Never voiced (T-0088a criterion 3): the path rewrite can spell one out of prose that held
        # none ("</eos>" -> "<eos>"). Unreachable from any other rewrite; main voiced it.
        text = re.sub(r"\s+", " ", drop_special_literals(text)).strip()
    return text, trimmed


def _last_token(text: str) -> str:
    """The run of non-whitespace characters ending at the end of ``text`` ("See e.g." → "e.g.")."""
    parts = text.split()
    return parts[-1] if parts else ""


def _is_balanced(chunk: str) -> bool:
    """True when ``chunk`` does not end inside a bracket or a code fence.

    A clause with an open bracket or an open fence is unfinished: it is held until more text closes
    it or ``close()`` releases it, never voiced early (criterion 3). Double quotes deliberately do
    not hold (A7): an unbalanced ``"`` is common in model prose (inch marks, an opening quote that
    closes sentences later) and holding on it defers first speech to ``llm.done``.
    """
    depth = 0
    for ch in chunk:
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSING_BRACKETS:
            depth = max(0, depth - 1)  # a closer for an earlier sentence's opener: ignore
    return depth == 0 and chunk.count("```") % 2 == 0


def _split(buf: str, *, final: bool) -> tuple[list[str], str]:
    """Split ``buf`` into complete sentences plus the held remainder.

    A boundary is a run of ``.``/``!``/``?`` (not an ellipsis, not an abbreviation, not a decimal)
    followed by whitespace — or by the end of the buffer, but only when ``final``: a terminator at
    a canvas edge may still grow into "..." or "3.14" on the next commit. Trailing whitespace is
    absorbed into the sentence that precedes it (A1), so the remainder starts at a word.
    """
    out: list[str] = []
    start = 0
    i = 0
    n = len(buf)
    while i < n:
        if buf[i] not in _TERMINATORS:
            i += 1
            continue
        end = i
        while end + 1 < n and buf[end + 1] in _TERMINATORS:
            end += 1
        if buf[i] == "." and end > i and set(buf[i : end + 1]) == {"."}:
            i = end + 1  # an ellipsis is not a sentence end
            continue
        k = end + 1
        while k < n and buf[k] in _CLOSERS:
            k += 1
        if k == n:
            if not final:
                break  # the next commit continues this token; hold it
        elif not buf[k].isspace():
            i = k  # "3.14", "a.b", "Four.<end_of_turn>": no boundary here
            continue
        if buf[end] == "." and _last_token(buf[start : end + 1]).lower() in _ABBREVIATIONS:
            i = k
            continue
        if not _is_balanced(buf[start:k]):
            i = k
            continue
        m = k
        while m < n and buf[m].isspace():
            m += 1
        out.append(buf[start:m])
        start = m
        i = m
    remainder = buf[start:]
    if final and remainder:
        out.append(remainder)
        remainder = ""
    return out, remainder


# --- the instruction stop (T-0232) ------------------------------------------------------------
#
# Found at James's desk on 2026-11-02 (turn t7): after 20 browser steps the model's answer was
# "Use no markdown, no lists, and numbers as words when short." and it was voiced. That is a
# paraphrase of the speaking sentence, so matching is on word sequences (ruling R-1), not on
# whole sentences.


def _instruction_texts() -> list[str]:
    """The fixed persona, speaking and tool guidance ``prompt_builder`` puts in the system message
    (criterion 1, ruling R-2), read from its constants at call time so nothing is copied here. The
    user's own instructions block, its header included (R-3), is not ours and is left out."""
    return [
        prompt_builder.CLOSE_THE_THOUGHT,
        prompt_builder.SPOKEN_STYLE,
        prompt_builder.NATIVE_TOOL_INSTRUCTIONS,
        *prompt_builder.LIVE_GUIDANCE.values(),
        prompt_builder.WAIT_TOOL_GUIDANCE,
    ]


def _instruction_sentences() -> tuple[str, ...]:
    """The instruction texts cut into sentences by this gate's own segmenter, stripped, unique."""
    out: list[str] = []
    for text in _instruction_texts():
        sentences, _ = _split(text, final=True)
        out.extend(s.strip() for s in sentences if s.strip())
    return tuple(dict.fromkeys(out))


#: T-0232 criterion 4: every sentence of the fixed system instructions, derived at import.
INSTRUCTION_SENTENCES: tuple[str, ...] = _instruction_sentences()

#: R-1: words dropped on both sides before matching, so "Use no markdown, ... and numbers" still
#: matches "no markdown, ..., numbers".
INSTRUCTION_FILLERS: frozenset[str] = frozenset(
    {"use", "and", "also", "please", "always", "then", "so", "you", "are", "i"}
)

#: R-7b: the forward floor. An answer sentence of fewer remaining words is never refused as a run
#: of an instruction sentence, so natural clauses such as "If it does not, try again." and "The
#: latest screenshot shows the page still loading." (8 words) stay voiced; the shortest logged
#: forward leak, the 2026-11-02 one, has 9.
INSTRUCTION_FORWARD_MIN_WORDS = 9

#: R-1: the reverse floor. An instruction sentence of fewer words is never looked for inside a
#: longer answer sentence.
INSTRUCTION_REVERSE_MIN_WORDS = 6

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _instruction_words(text: str) -> tuple[str, ...]:
    """R-1's word sequence: lowercase, ``[a-z0-9_]+`` words, fillers dropped."""
    return tuple(w for w in _WORD_RE.findall(text.lower()) if w not in INSTRUCTION_FILLERS)


#: Each instruction sentence's word sequence, as written and as spoken (``google_search`` is one
#: word in the prompt, two in the spoken text the candidate is compared in).
_INSTRUCTION_SEQUENCES: frozenset[tuple[str, ...]] = frozenset(
    seq
    for sentence in INSTRUCTION_SENTENCES
    for seq in (_instruction_words(sentence), _instruction_words(normalise_for_tts(sentence)))
)


def _is_run(needle: tuple[str, ...], hay: tuple[str, ...]) -> bool:
    """``needle`` is a contiguous run of ``hay``."""
    n = len(needle)
    return any(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def _is_instruction(text: str) -> bool:
    """T-0232 R-1: the spoken ``text`` voices an instruction sentence. Forward: its word sequence
    (at least :data:`INSTRUCTION_FORWARD_MIN_WORDS`, R-7b) is a contiguous run of one instruction
    sentence's. Reverse (self-talk quoting the instructions): it contains one instruction
    sentence's whole sequence (at least :data:`INSTRUCTION_REVERSE_MIN_WORDS`) contiguously. Pure
    and total."""
    candidate = _instruction_words(text)
    for seq in _INSTRUCTION_SEQUENCES:
        if len(candidate) >= INSTRUCTION_FORWARD_MIN_WORDS and _is_run(candidate, seq):
            return True
        if len(seq) >= INSTRUCTION_REVERSE_MIN_WORDS and _is_run(seq, candidate):
            return True
    return False


@dataclass
class SpeechGate:
    pending: str = field(default="", init=False)
    emitted: list[Sentence] = field(default_factory=list, init=False)
    #: Sentences the voice guard refused this turn (criterion 10). The count is every refusal;
    #: the sources stop at ``MAX_REFUSED_SOURCES``. Neither is reset by ``close()``: one gate is
    #: one turn, and the turn's record is read after it ends (publishing them is T-0054).
    refused_count: int = field(default=0, init=False)
    refused_sources: list[str] = field(default_factory=list, init=False)
    #: Why each retained source was refused, index for index with ``refused_sources``:
    #: ``"interface_data"`` (the T-0052 voice guard), ``"repeat"`` (T-0224 criterion 1) or
    #: ``"instructions"`` (T-0232).
    silenced_reasons: list[str] = field(default_factory=list, init=False)
    _stopped: bool = field(default=False, init=False)  # an EOS/think token has been seen
    #: T-0224: the repeat keys of the sentences voiced this turn, and whether a repeat has
    #: stopped the turn's speech. Both reset by ``close()``, like ``_stopped``.
    _voiced_keys: list[str] = field(default_factory=list, init=False)
    _repeat_stopped: bool = field(default=False, init=False)

    def feed(self, commit: Commit) -> Sequence[Sentence]:
        """Feed one committed answer chunk; return the sentences ready to speak now."""
        if commit.segment != "answer":
            raise GateInputError(
                f"speech gate accepts answer segments only, got {commit.segment!r}"
            )
        if commit.thought_open:
            raise GateInputError("canvas ends inside an unclosed thought; never voice it")
        return self._consume(self.pending + commit.text, final=False)

    def close(self) -> Sequence[Sentence]:
        """``llm.done``: flush the held clause. Calling it again speaks nothing again."""
        out = self._consume(self.pending, final=True)
        self._stopped = False  # the turn is over; a gate reused for the next one starts clean
        self._voiced_keys = []
        self._repeat_stopped = False
        return out

    def _consume(self, buf: str, *, final: bool) -> Sequence[Sentence]:
        sources, self.pending = _split(buf, final=final)
        sentences: list[Sentence] = []
        for source in sources:
            sentence = self._sentence(source)  # built first: a refusal leaves the same state
            if _is_interface_data(source):
                sentence = self._refuse_interface_data(sentence)
            elif sentence.text and not self._repeat_stopped and _is_instruction(sentence.text):
                # T-0232: no latch, and not added to the voiced keys (it was not voiced). After a
                # repeat latch the branch below reports "repeat" instead (R-4).
                sentence = self._refuse(sentence, "instructions")
            elif sentence.text and (self._repeat_stopped or self._is_repeat(sentence.text)):
                self._repeat_stopped = True
                sentence = self._refuse(sentence, "repeat")
            elif sentence.text:
                self._voiced_keys.append(_repeat_key(sentence.text))
            sentences.append(sentence)
        self.emitted.extend(sentences)
        return sentences

    def _is_repeat(self, text: str) -> bool:
        """T-0224 criterion 1: ``text`` re-says a sentence already voiced this turn.

        A voiced key of at least :data:`REPEAT_MIN_CHARS` that equals the candidate's key, or
        sits inside it as a whole sentence: it starts at a word boundary and ends either at the
        end of the candidate or just before a ``.``/``!``/``?`` inside it (ruling R-7). The
        containment half is what catches P4 r1's u9 ("All good.You have a mistake in the formula
        for acceleration.") and t0212's "I'll click the first rotation value.I will click on …":
        a repeat glued to its neighbour with no space for the segmenter to split on. The sentence
        end is what keeps a correction voiced: "Check: your answer is forty two point five."
        contains "your answer is forty two" but not as a sentence.
        """
        key = _repeat_key(text)
        for voiced in self._voiced_keys:
            if len(voiced) < REPEAT_MIN_CHARS:
                continue
            start = key.find(voiced)
            while start != -1:
                end = start + len(voiced)
                if (start == 0 or not key[start - 1].isalnum()) and (
                    end == len(key) or key[end] in _TERMINATORS
                ):
                    return True
                start = key.find(voiced, start + 1)
        return False

    def _refuse_interface_data(self, sentence: Sentence) -> Sentence:
        """Criteria 10 and 11: silence it, count it, keep it retrievable, change nothing else.

        **Conserve, do not drop (ruling A20).** The refused sentence comes back with its ``source``
        byte-exact and ``text=""`` — the same idiom :meth:`_sentence` uses for post-EOS bytes just
        below. The turn machine checks ``"".join(sources) + remainder == text`` on every segmenter
        call and, when it fails, raises ``ValueError`` and emits *no* ``tts.speak`` at all, which
        ``session.py`` then swallows without logging; measured end to end, dropping the item turned
        "the human hears 807 characters of JSON" into a silent turn with nothing in the log and one
        probe stuck in SPEAKING for 20 000 ms. An empty ``text`` voices nothing, so the safety rule
        and the conservation law are both satisfied.

        The text is dropped rather than summarised (T-0052 D5): the surface is already on screen
        and the model's own prose usually says what it built. Nothing is printed, logged or
        raised — this module is pure; the caller reads ``refused_count``/``refused_sources``. The
        pending buffer and the EOS latch are exactly as they would be had the sentence been
        spoken, so a refusal loses that sentence and never the rest of the answer.
        """
        return self._refuse(sentence, "interface_data")

    def _refuse(self, sentence: Sentence, reason: str) -> Sentence:
        """The one refusal path (T-0052 A20, reused by T-0224's repeat stop): count it, keep its
        source and reason retrievable, and conserve it with ``text=""``."""
        self.refused_count += 1
        if len(self.refused_sources) < MAX_REFUSED_SOURCES:
            self.refused_sources.append(sentence.source)
            self.silenced_reasons.append(reason)
        return Sentence(source=sentence.source, text="")

    def _sentence(self, source: str) -> Sentence:
        """A2/A3: post-EOS bytes travel in ``source`` but are never voiced."""
        spoken, trimmed = _normalise(source)
        text = "" if self._stopped else spoken
        if trimmed:
            self._stopped = True
        return Sentence(source=source, text=text)
