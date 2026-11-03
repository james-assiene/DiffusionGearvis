"""Structural (bracket) repair of an A2UI canvas before json_repair sees it. Task T-0108.

The model's A2UI JSON goes wrong in a few structural ways that json_repair turns into junk
(docs/tasks/T-0108.md *Goal*). :func:`repair_structure` undoes exactly those, in one
left-to-right pass that is string-aware (escapes respected), keeps an explicit stack of open
containers and looks ahead a bounded number of characters (plus, at a mismatched closer, the
whitespace run after it). No recursion, no backtracking regex; linear in the input.

Edit rules, each decided outside a string:

1. **Duplicated opener** ``{"{"id":`` → ``{"id":``: right after an object's ``{``, a key ``"{"``
   that is followed by the start of a key name (not ``:``) — the ``"{`` is deleted.
2. **Missing separator** ``"From:" "{"id":`` → ``"From:"},{"id":``: in an object of an array, a
   ``"{"`` + key name where ``,`` or ``}`` was due — the object is closed and the next one opened.
3. **Stray quote** ``["a"]"]},`` → ``["a"]]},`` (then rule 4): in an object, a ``"`` right after
   a container value and right before a closer is deleted.
4. **Closers.** A closer that does not match the innermost open container is dropped when the
   expected closer follows (after whitespace); otherwise the expected closer(s) are inserted
   until it matches. A closer with no opener of its kind is dropped.

Only the characters ``{ } [ ] , "`` are inserted or deleted, never inside a string, so every
string lexeme of the output is one of the input's. At end of text nothing is closed (a partial
canvas stays partial; json_repair closes truncations). Text that is already valid JSON triggers
no rule, so it comes back byte-identical with 0 edits.
"""

from __future__ import annotations

import re

__all__ = ["repair_structure"]

#: the characters the scanner acts on outside a string
_STRUCTURAL = re.compile(r'[{}\[\]",:]')
#: what ends or escapes inside a string
_IN_STRING = re.compile(r'["\\]')
_JSON_WS = " \t\n\r"
_NON_WS = re.compile(r"[^ \t\n\r]")
#: a character that cannot begin a key name, for rules 1 and 2
_NOT_KEY_START = frozenset('{}[],:"' + _JSON_WS)

_CLOSER = {"{": "}", "[": "]"}
_OPENER = {"}": "{", "]": "["}

# frame states: an object right after ``{``, after ``,``, after a key, after ``:``; anything
# (object or array) after a complete value; an array expecting a value
_KEY0, _KEY, _COLON, _VALUE, _AFTER = range(5)


def repair_structure(text: str) -> tuple[str, int]:
    """``(repaired text, number of structural edits)``; ``(text, 0)`` when no rule applies."""
    n = len(text)
    out: list[str] = []
    start = 0  # text[start:i] is still to be copied to ``out``
    edits = 0
    kinds: list[str] = []
    states: list[int] = []
    opened = {"{": 0, "[": 0}
    after_container = False  # the last token outside a string was a closer
    i = 0
    while True:
        match = _STRUCTURAL.search(text, i)
        if match is None:
            break
        j = match.start()
        if j > i and text[i:j].strip(_JSON_WS):
            # a scalar (number, literal, or junk) between structural characters
            if states and states[-1] == _VALUE:
                states[-1] = _AFTER
            after_container = False
        c = text[j]
        in_object = bool(kinds) and kinds[-1] == "{"

        if c == '"':
            state = states[-1] if states else _AFTER
            if (
                in_object
                and state in (_KEY0, _AFTER)
                and text.startswith('"{"', j)
                and j + 3 < n
                and text[j + 3] not in _NOT_KEY_START
                and (state == _KEY0 or (len(kinds) >= 2 and kinds[-2] == "["))
            ):
                out.append(text[start:j])
                if state == _AFTER:
                    # rule 2: close this object, open the next one in the same array
                    out.append("},{")
                    states[-1] = _KEY0
                # rule 1 (and the tail of rule 2): drop ``"{``; the key string starts at j + 2
                start = i = j + 2
                edits += 1
                after_container = False
                continue
            if (
                in_object
                and state == _AFTER
                and after_container
                and j + 1 < n
                and text[j + 1] in "}]"
            ):
                out.append(text[start:j])  # rule 3
                start = i = j + 1
                edits += 1
                continue
            end = _string_end(text, j + 1)
            if end < 0:
                break  # the text ends inside this string: nothing more to decide
            if states:
                if in_object:
                    if state in (_KEY0, _KEY):
                        states[-1] = _COLON
                    elif state == _VALUE:
                        states[-1] = _AFTER
                elif state == _VALUE:
                    states[-1] = _AFTER
            after_container = False
            i = end + 1
            continue

        i = j + 1
        if c == ":":
            if in_object and states[-1] == _COLON:
                states[-1] = _VALUE
            after_container = False
        elif c == ",":
            if states:
                states[-1] = _KEY if in_object else _VALUE
            after_container = False
        elif c in "{[":
            if states:
                states[-1] = _AFTER
            kinds.append(c)
            states.append(_KEY0 if c == "{" else _VALUE)
            opened[c] += 1
            after_container = False
        else:  # a closer
            opener = _OPENER[c]
            if kinds and kinds[-1] == opener:
                kinds.pop()
                states.pop()
                opened[opener] -= 1
                after_container = True
                continue
            if kinds and opened[opener] > 0 and not _next_is(text, j + 1, _CLOSER[kinds[-1]]):
                # rule 4, insert: close what is open until this closer matches
                out.append(text[start:j])
                while kinds[-1] != opener:
                    inner = kinds.pop()
                    states.pop()
                    opened[inner] -= 1
                    out.append(_CLOSER[inner])
                    edits += 1
                kinds.pop()
                states.pop()
                opened[opener] -= 1
                start = j
                after_container = True
                continue
            # rule 4, drop: the expected closer follows, or nothing of this kind is open
            out.append(text[start:j])
            start = j + 1
            edits += 1
    if edits == 0:
        return text, 0
    out.append(text[start:])
    return "".join(out), edits


def _string_end(text: str, k: int) -> int:
    """Index of the ``"`` closing the string whose content starts at ``k``; -1 if unterminated."""
    while True:
        match = _IN_STRING.search(text, k)
        if match is None:
            return -1
        k = match.start()
        if text[k] == '"':
            return k
        k += 2  # an escape: skip the escaped character, whatever it is


def _next_is(text: str, k: int, expected: str) -> bool:
    """Whether the first non-whitespace character at or after ``k`` is ``expected``."""
    match = _NON_WS.search(text, k)
    return match is not None and text[match.start()] == expected
