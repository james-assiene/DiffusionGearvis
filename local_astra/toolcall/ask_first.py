"""T-0107a: the ask-first backstop's predicate (pure, stdlib only, total, linear time).

The model judges when a step is high-stakes and asks (H-T107); the conductor only makes the model's own question
binding. ``asks_user(text)`` is true iff some ``?`` (U+003F or U+FF1F) satisfies both:

(a′) the character right after it, if any, is neither lowercase (``str.islower()``) nor a digit (``str.isdigit()``),
    so ``3?4`` and ``pass?word`` do not ask but the glued ``Shall I pay?Then`` and ``Mona Lisa?Leonardo`` do
    (T-0107d);
(b) after skipping every following character that is not alphanumeric, either the end of the text is reached, or
    the first alphanumeric character is not lowercase (``not ch.islower()``: uppercase, titlecase, a digit, ...).

So ``Shall I pay?``, ``**Shall I pay?**`` and ``Proceed?\\n1. Yes`` ask; the label ``the Forgot password? link``
does not (a lowercase word follows). Anything may precede the mark, so French spacing (``commande ?``) asks.

One right-to-left pass: the first alphanumeric after each index is carried as state, so the scan is linear.
"""

from __future__ import annotations

#: the reason a call drafted in an asking step is refused with (a constant: never text built from the call)
ASKED_FIRST_REASON = "asked the user first"

_MARKS = frozenset("?？")


def asks_user(text: str) -> bool:
    """True iff ``text`` holds a question mark that asks (see the module docstring); never raises."""
    # next_is_lower: whether the first alphanumeric strictly after the current index is lowercase (False if none)
    next_is_lower = False
    # following_is_lower_or_digit: whether the character right after the current index is lowercase or a digit
    following_is_lower_or_digit = False
    for ch in reversed(text):
        if ch in _MARKS and not following_is_lower_or_digit and not next_is_lower:
            return True
        if ch.isalnum():
            next_is_lower = ch.islower()
        following_is_lower_or_digit = ch.islower() or ch.isdigit()
    return False
