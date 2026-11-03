"""The announced-call rule and the re-ask prompt (T-0172 group A, docs/tasks/T-0172.md A1, A2).

A tools-on step sometimes ends with only a promise ("I will search for the current weather in Paris again.") and no
call. ``announces`` recognises that shape; ``reask_prompt`` builds the one extra request the session makes then.

The rule (A1), case-insensitive, on the stripped text:

- size: non-empty, at most ``MAX_CHARS`` characters and ``MAX_SENTENCES`` sentences (split on ``.``, ``!``, ``?``
  followed by whitespace or the end, and on newlines);
- no ``?`` anywhere, no conditional addressed to the user (``if you``, ``when you``, ``once you``, ``should you``)
  and no ``let me know`` anywhere;
- intent: the last sentence (``tail``) or every sentence (``all``) is an optional lead-in from a closed list, a
  first-person future subject, an optional adverb, an optional ``try to``/``try``, then a verb in ``ACTION_VERBS``
  and not in ``DENY_VERBS``;
- A1b, on the matched sentence: no non-actionable particle after the verb (``go with``, ``look into``, ``check
  back``, ``try my best``, ...), no colon right after it, no deferral after it (``later``, ``tomorrow``, ``next
  time``, ``another time``, ``soon``), and the user is not its object (``with you``, ``for you first``, ``ask you``);
- A1c: ``go ahead and <verb>`` is judged as ``<verb>``; ``go ahead`` alone stays a refused particle.

Pure: standard library and ``local_astra.contracts`` only; linear in the text; never raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from local_astra.contracts.llm import Message, Prompt, TextPart

#: A1 size bounds, on the stripped text.
MAX_CHARS = 240
MAX_SENTENCES = 3

#: Verbs of an action that needs a tool (a search, or an action on the screen).
ACTION_VERBS: frozenset[str] = frozenset(
    {
        "search",
        "look",
        "check",
        "find",
        "click",
        "tap",
        "press",
        "open",
        "go",
        "visit",
        "navigate",
        "browse",
        "scroll",
        "type",
        "enter",
        "fill",
        "select",
        "deselect",
        "add",
        "remove",
        "refresh",
        "reload",
        "submit",
        "fetch",
        "retry",
        "try",
    }
)

#: Verbs of a promise that is no action (remembering, waiting, telling the user).
DENY_VERBS: frozenset[str] = frozenset(
    {
        "remember",
        "keep",
        "note",
        "wait",
        "tell",
        "explain",
        "answer",
        "provide",
        "know",
        "think",
        "be",
    }
)

#: The user turn appended to the re-ask (A2). Never stored in history, never published.
REASK_NUDGE = (
    "You said what you would do next but made no tool call. Make that tool call now. "
    "If something stops you, tell the user in one sentence what stopped you instead."
)

Mode = Literal["tail", "all"]


@dataclass(frozen=True)
class Announcement:
    """A hit: the action ``verb`` (lower case) and the ``sentence`` that announced it."""

    verb: str
    sentence: str


_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|\n")
_USER_CONDITION = re.compile(
    r"\b(?:if|when|once|should)\s+you\b|\blet\s+me\s+know\b", re.IGNORECASE
)
_INTENT = re.compile(
    r"(?:(?:now|next|actually|okay|so|then|first)\b,?\s*)?"
    r"(?:i\s+will|i['’]ll|let\s+me|i\s+am\s+going\s+to|i['’]m\s+going\s+to|i\s+shall)\s+"
    r"(?:(?:now|first|then|also|just)\s+)?"
    r"(?:go\s+ahead\s+and\s+)?"  # A1c: "go ahead and <verb>" is judged as "<verb>"
    r"(?:(?P<try>try)(?P<to>\s+to)?\s+)?"
    r"(?P<verb>[a-z]+)\b",
    re.IGNORECASE,
)


#: A1b: particles after the verb that make it no action ("I'll go with option two.", "I'll try my best.").
_PHRASAL: dict[str, re.Pattern[str]] = {
    "go": re.compile(r"\s+(?:with|ahead|over|through)\b", re.IGNORECASE),
    "look": re.compile(r"\s+(?:forward|into)\b", re.IGNORECASE),
    "check": re.compile(r"\s+(?:back|with)\b", re.IGNORECASE),
    "try": re.compile(r"\s+(?:my\s+best|harder|again\s+later)\b", re.IGNORECASE),
}
#: A1b: a deferral after the verb ("I'll check back later.", "I will search for flights tomorrow.").
_DEFERRAL = re.compile(r"\b(?:later|tomorrow|next\s+time|another\s+time|soon)\b", re.IGNORECASE)
#: A1b: the user as the object ("Let me check with you first.", "... and ask you to confirm.").
_USER_OBJECT = re.compile(r"\b(?:with\s+you|for\s+you\s+first|ask\s+you)\b", re.IGNORECASE)
#: A1b: a colon right after the verb: the answer is delivered inline ("Let me check: it is 20 degrees.").
_COLON = re.compile(r"\s*:")


def _sentences(text: str) -> list[str]:
    """The non-empty sentences of ``text``, stripped, each with its terminator."""
    out: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        piece = text[start : m.end()].strip()
        start = m.end()
        if piece.rstrip(".!?").strip():
            out.append(piece)
    piece = text[start:].strip()
    if piece.rstrip(".!?").strip():
        out.append(piece)
    return out


def _intent(sentence: str) -> Announcement | None:
    m = _INTENT.match(sentence)
    if m is None:
        return None
    verb = m.group("verb").lower()
    verb_end = m.end("verb")
    if m.group("try") is not None and verb not in ACTION_VERBS:
        if m.group("to") is not None or verb in DENY_VERBS:
            return None  # "try to <not an action>", "try <denied>"
        verb = "try"  # "I will try clicking ...", "I will try one more time."
        verb_end = m.end("try")
    if verb not in ACTION_VERBS or verb in DENY_VERBS:
        return None
    rest = sentence[verb_end:]
    phrasal = _PHRASAL.get(verb)
    if phrasal is not None and phrasal.match(rest):
        return None
    if _COLON.match(rest) or _DEFERRAL.search(rest) or _USER_OBJECT.search(sentence):
        return None
    return Announcement(verb=verb, sentence=sentence)


def announces(text: str, *, mode: Mode = "tail") -> Announcement | None:
    """The announcement ``text`` ends with (A1), or ``None``. ``mode="all"`` needs every sentence to announce."""
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_CHARS or "?" in stripped:
        return None
    if _USER_CONDITION.search(stripped):
        return None
    sentences = _sentences(stripped)
    if not sentences or len(sentences) > MAX_SENTENCES:
        return None
    last = _intent(sentences[-1])
    if last is None:
        return None
    if mode == "all" and any(_intent(s) is None for s in sentences[:-1]):
        return None
    return last if mode in ("tail", "all") else None


def reask_prompt(sent: Prompt, said: str) -> Prompt:
    """The re-ask request (A2): ``sent`` with ``said`` as the model's words and ``REASK_NUDGE`` as a user turn.

    Step 1 (``sent`` opens a model turn): ``said`` becomes a new assistant message. Step >= 2 (``sent`` continues a
    final assistant message): that message's text is followed by ``said``. Every other field is ``sent``'s; the
    result never continues the final message, so the template opens a fresh model turn after the nudge.
    """
    fields = {f: getattr(sent, f) for f in Prompt.model_fields}
    if sent.continue_final_message:
        head = sent.messages[:-1]
        said = sent.messages[-1].text() + said
    else:
        head = sent.messages
    fields["messages"] = (
        *head,
        Message(role="assistant", content=(TextPart(text=said),)),
        Message(role="user", content=(TextPart(text=REASK_NUDGE),)),
    )
    fields["continue_final_message"] = False
    return Prompt(**fields)
