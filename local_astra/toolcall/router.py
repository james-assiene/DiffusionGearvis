"""The router (T-0080a criterion 2): find the mode tag in a committed answer. Pure and total.

``route(answer_text, *, thought_open=False) -> Route``. Only the answer channel is scanned:

- with ``thought_open`` set (``llm_client``'s own flag), or with a ``<|channel>`` that no later
  ``<channel|>`` closes, the route is ``none`` and nothing is spoken (``spoken_prefix == ""``): text
  from inside a thought is never voiced and never routes;
- closed thought blocks ``<|channel>...<channel|>`` are removed before the scan, so a tag inside a
  thought never routes and thought text never reaches ``spoken_prefix``;
- the first tag of ``TAGS`` in what remains wins. ``<cc>`` and the native ``<|tool_call>`` route
  ``toolcall`` (``tag="cc"`` / ``"native"``), ``<genui>`` routes ``genui`` with no draft parsing;
- T-0085 (amending T-0080a/T-0082a criterion 2): ``spoken_prefix`` is the scanned text before the tag (or all of
  it for ``none``) with every channel marker deleted, repeatedly, until none is left (a stray ``<channel|>`` after
  the spoken sentence would otherwise be voiced and break the T4 continuation, since the pinned template's
  ``strip_thinking`` deletes it). The tag is found *before* the removal, so a removal never creates a tag. Drafts
  are untouched.

For ``cc`` only, a terminal close after the call's brace (the text after ``<cc>`` matches
``}\\s*</cc>\\s*\\Z``) is moved out of the draft into ``draft_close`` (``</cc>`` plus the whitespace after it;
whitespace before it stays in the draft); otherwise ``draft_close == ""`` (T-0082a, amending T-0080a
criterion 2). ``TAGS[tag] + draft_call_text + draft_close`` is exactly the scanned text from the tag onward.
``template_content(text)`` mirrors the pinned chat template's rendering of assistant string content (its
``strip_thinking`` macro); its truth is the oracle test against the pinned template, not a runtime jinja2 render.

``draft_name`` is the tool a JSON draft ``{"name": ...}`` names (read from the stripped draft; also when the
draft is truncated after the name), only for ``cc`` and only if that tool is in the v1 set; otherwise ``None``.

Adding a mode is a row in ``TAGS`` and ``KIND_OF_TAG`` plus a handler in the conductor.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from typing import Literal

from local_astra.toolcall import tools

__all__ = [
    "CHANNEL_CLOSE",
    "CHANNEL_OPEN",
    "KIND_OF_TAG",
    "TAGS",
    "Kind",
    "Route",
    "answer_channel",
    "route",
    "template_content",
]

Kind = Literal["none", "toolcall", "genui"]

CHANNEL_OPEN = "<|channel>"
CHANNEL_CLOSE = "<channel|>"

#: Tag name -> its literal text. Closed mapping (umbrella T-0080, "Room for GenUI").
TAGS: dict[str, str] = {"cc": "<cc>", "genui": "<genui>", "native": "<|tool_call>"}
KIND_OF_TAG: dict[str, Kind] = {"cc": "toolcall", "genui": "genui", "native": "toolcall"}

_DRAFT_NAME = re.compile(r'\A\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"')
#: T-0082a: a terminal ``</cc>`` plus trailing whitespace (group 1) right after the call's ``}`` (+ whitespace).
_TERMINAL_CLOSE = re.compile(r"\}\s*(</cc>\s*)\Z")


@dataclass(frozen=True)
class Route:
    kind: Kind
    tag: str | None
    spoken_prefix: str
    draft_call_text: str
    draft_name: str | None
    draft_close: str = ""


@functools.cache
def _v1_names() -> frozenset[str]:
    return frozenset(tools.load().names)


def answer_channel(text: str) -> tuple[str, bool]:
    """T-0117: the answer channel of ``text`` and whether a thought was left open.

    Each closed block ``<|channel>...<channel|>`` is removed with its markers (the first ``<channel|>`` after an
    open closes it); the text from a ``<|channel>`` that nothing after it closes is cut, and the flag says so.
    Nothing is inserted: the bytes around a removed block are kept exactly. One pass, no literal deletion."""
    out: list[str] = []
    pos = 0
    while True:
        i = text.find(CHANNEL_OPEN, pos)
        if i < 0:
            out.append(text[pos:])
            return "".join(out), False
        j = text.find(CHANNEL_CLOSE, i + len(CHANNEL_OPEN))
        out.append(text[pos:i])
        if j < 0:
            return "".join(out), True
        pos = j + len(CHANNEL_CLOSE)


def _strip_closed_thoughts(text: str) -> str | None:
    """The answer channel of ``text``; ``None`` if a thought is left open."""
    channel, cut = answer_channel(text)
    return None if cut else channel


def _drop_channel_markers(text: str) -> str:
    """``text`` with channel markers deleted until none is left (a deletion can splice a new one)."""
    while CHANNEL_OPEN in text or CHANNEL_CLOSE in text:
        text = text.replace(CHANNEL_CLOSE, "").replace(CHANNEL_OPEN, "")
    return text


def template_content(text: str) -> str:
    """The pinned template's ``strip_thinking(text)``: split on ``<channel|>``, keep each piece up to its first
    ``<|channel>``, join, trim (jinja's ``trim`` is ``str.strip``)."""
    return "".join(part.split(CHANNEL_OPEN)[0] for part in text.split(CHANNEL_CLOSE)).strip()


def _draft_name(draft: str) -> str | None:
    name: object = None
    try:
        obj = json.loads(draft)
    except (ValueError, RecursionError):  # not JSON (or pathologically nested): read the prefix
        m = _DRAFT_NAME.match(draft)
        if m is not None:
            try:
                name = json.loads('"' + m.group(1) + '"')
            except ValueError:
                name = None
    else:
        if isinstance(obj, dict):
            name = obj.get("name")
    if isinstance(name, str) and name in _v1_names():
        return name
    return None


def route(answer_text: str, *, thought_open: bool = False) -> Route:
    """Route a committed answer. Never raises on any ``str`` input."""
    scanned = None if thought_open else _strip_closed_thoughts(answer_text)
    if scanned is None:
        return Route(kind="none", tag=None, spoken_prefix="", draft_call_text="", draft_name=None)
    first: tuple[int, str] | None = None
    for tag, literal in TAGS.items():
        i = scanned.find(literal)
        if i >= 0 and (first is None or i < first[0]):
            first = (i, tag)
    if first is None:
        return Route(
            kind="none",
            tag=None,
            spoken_prefix=_drop_channel_markers(scanned),
            draft_call_text="",
            draft_name=None,
        )
    i, tag = first
    draft = scanned[i + len(TAGS[tag]) :]
    close = ""
    if tag == "cc":
        m = _TERMINAL_CLOSE.search(draft)
        if m is not None:
            draft, close = draft[: m.start(1)], m.group(1)
    return Route(
        kind=KIND_OF_TAG[tag],
        tag=tag,
        spoken_prefix=_drop_channel_markers(scanned[:i]),
        draft_call_text=draft,
        draft_name=_draft_name(draft) if tag == "cc" else None,
        draft_close=close,
    )
