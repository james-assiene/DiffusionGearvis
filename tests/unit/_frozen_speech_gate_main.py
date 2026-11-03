# Frozen copy of local_astra/conductor/speech_gate.py at main 575e1d4 (T-0088a differential oracle).
# Never edit: tests/unit/test_speech_gate_special_tokens.py compares the T-0088a gate with this on
# committed text that contains no Gemma 4 special-token literal (criterion 3, negative).
# ruff: noqa
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
  ``text=""`` (ruling A20), so only the *voicing* is lost.

Rulings implemented here (docs/tasks/T-0004.md amendments):

- **A1, conservation is byte-exact.** ``"".join(source) + pending`` rebuilds every byte fed so far,
  at every step — the turn machine (T-0003 criterion 10) raises ``ValueError`` otherwise. The
  whitespace following a terminator rides on the *preceding* sentence's ``source``; ``text`` is
  that source normalised (which strips it), so ``pending`` always starts at a word.
- **A2/A3, special tokens.** An EOS token (``<end_of_turn>``, ``<eos>``, ``<|endoftext|>``) means
  the model stopped: everything from it on is trimmed out of ``text``. ``<|think|>`` trims too —
  it switches the model back to the thought channel, and thinking is never voiced (CLAUDE.md).
  Any other ``<|…|>`` token is a tokenizer artifact: it is dropped in place and the surrounding
  text joined. The trimmed/dropped bytes still travel inside a ``source`` (conservation) with a
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

from local_astra.contracts.llm import Commit


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
#: stopped (EOS) or gone back to thinking (``<|think|>``), and neither is answer text.
_TRIM_TOKENS: tuple[str, ...] = ("<end_of_turn>", "<eos>", "<|endoftext|>", "<|think|>")

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


def normalise_for_tts(text: str) -> str:
    """File paths, code identifiers, fenced code and special tokens → speakable text.

    Deliberately narrow (see module docstring); identity on ordinary prose, numbers, dates and
    URLs. Leading/trailing and repeated whitespace is collapsed: ``text`` is what reaches the TTS,
    not what the model committed (``source`` carries that, byte for byte).
    """
    cut = _trim_index(text)
    if cut is not None:
        text = text[:cut]
    text = _FENCE_RE.sub(" a code block. ", text)
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    text = _PATH_RE.sub(_say_path, text)
    text = _IDENT_RE.sub(_say_identifier, text)
    return re.sub(r"\s+", " ", text).strip()


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


@dataclass
class SpeechGate:
    pending: str = field(default="", init=False)
    emitted: list[Sentence] = field(default_factory=list, init=False)
    #: Sentences the voice guard refused this turn (criterion 10). The count is every refusal;
    #: the sources stop at ``MAX_REFUSED_SOURCES``. Neither is reset by ``close()``: one gate is
    #: one turn, and the turn's record is read after it ends (publishing them is T-0054).
    refused_count: int = field(default=0, init=False)
    refused_sources: list[str] = field(default_factory=list, init=False)
    _stopped: bool = field(default=False, init=False)  # an EOS/think token has been seen

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
        return out

    def _consume(self, buf: str, *, final: bool) -> Sequence[Sentence]:
        sources, self.pending = _split(buf, final=final)
        sentences: list[Sentence] = []
        for source in sources:
            sentence = self._sentence(source)  # built first: a refusal leaves the same state
            if _is_interface_data(source):
                sentence = self._refuse_interface_data(sentence)
            sentences.append(sentence)
        self.emitted.extend(sentences)
        return sentences

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
        self.refused_count += 1
        if len(self.refused_sources) < MAX_REFUSED_SOURCES:
            self.refused_sources.append(sentence.source)
        return Sentence(source=sentence.source, text="")

    def _sentence(self, source: str) -> Sentence:
        """A2/A3: post-EOS bytes travel in ``source`` but are never voiced."""
        text = "" if self._stopped else normalise_for_tts(source)
        if _trim_index(source) is not None:
            self._stopped = True
        return Sentence(source=source, text=text)
