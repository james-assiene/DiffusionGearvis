"""The tool-result loop's host-side rendering (T-0113; T-0111a §1 form G, §4; H-T111a-9, H-T111a-10). Pure.

Each loop step is a new request: the step-1 prompt as sent plus one final ``assistant`` message, continued
(``continue_final_message``), whose text is ``text_1 + call_1 + response_1 + text_2 + call_2 + response_2 ...``.
``render_call`` and ``render_response`` reproduce the pinned Gemma 4 template (sha ``36e3a42e...``):
``format_argument`` (lines 118-147, ``escape_keys=False``), the ``tool_calls`` loop (lines 243-258) and
``format_tool_response_block`` (lines 160-173). Two template facts a naive renderer misses: ``dictsort`` is
case-insensitive, and a boolean renders as ``true``/``false``.

**Sanitiser.** Special-token literals in assistant text are tokenized as special ids (T-0087 q9), so every string
of a call, of a response and of a step text passes ``drop_special_literals`` (to a fixpoint) before it is embedded;
the only literals left are the renderer's own frame and ``<|"|>`` delimiters.

**Bounds.** ``search_payload`` keeps at most ``MAX_RESULTS`` results and cuts each snippet to ``MAX_SNIPPET``
chars. ``render_response`` bounds the whole ``<|tool_response>...<tool_response|>`` block to ``MAX_RESPONSE``
chars by capping every string at the largest common length that fits (a cut string ends in ``…``).

**Compaction (T-0120; H-T115-0, H-T120-1..3).** ``compact(steps, budget)`` returns the steps unchanged when their
continuation fits ``budget``; otherwise it replaces older responses (oldest first, never the last ``keep_last``) with
``elided_payload`` stubs until the continuation fits ``target`` (default ``budget * 2 // 3``), or every candidate is
elided. Calls and step texts are kept. A search stub invites searching again (``ELIDED_SEARCH_NOTE``, H-T120b-0); an
action's or a plain stub only says it was omitted (``ELIDED_NOTE``). The session calls it before its over-budget
check and keeps the compacted steps (T-0120b).

**The live screen (T-0111g B8).** ``swap_newest_image(prompt, frame)`` replaces the newest ``ImagePart`` of the
prompt's final user message (the current turn) with the part the prompt builder makes for ``frame``; a current turn
with no image comes back equal, and a history image is never swapped. ``show_current_image`` is what the session
sends: the swap, or, for a current turn whose frame rides on the newest history image, the frame placed first in
the current user message. ``screenshot_payload`` adds
``"screenshot": "updated" | "unchanged"`` to a live step's response.

**The report step (T-0178 B1, H-T178-3).** ``with_report_fact(prompt)`` appends ``REPORT_SEPARATOR + REPORT_FACT``
to a step request's continuation, after the last tool response and outside any payload; every earlier byte is kept.
The session sends it on the report step only (the step after the last allowed call).

**Screen reminder (T-0201 B, H-T201-1..4).** ``with_screen_reminder(prompt, reminder)`` appends
``SCREEN_REMINDER_SEPARATOR + reminder`` to a step request's continuation, after the last tool response and outside
any payload, the same append-only way: the continuation is not interpreted, sanitised or closed, and every other
field, message and part is kept. The text after the continuation is the last thing the model reads, so a labelled
live step tells it there, too, that the current screenshot is in the user's message above (the session picks the
reminder by the label it placed; ``conductor.session.SCREEN_REMINDERS``). It lives only in the request: the steps the
session keeps never hold it, so step k+1 extends step k's bytes without step k's reminder. Prefix-cache cost: the
cached prefix of step k+1 ends where step k's reminder began, so at most the reminder's tokens plus one KV block are
recomputed per step.

Imports: the stdlib, ``local_astra.contracts`` and ``toolcall.special_tokens`` only (criterion 13).
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from local_astra.contracts.llm import ImagePart, Message, Prompt, TextPart
from local_astra.contracts.perception import Frame
from local_astra.contracts.toolcall import SearchResults
from local_astra.toolcall.special_tokens import drop_special_literals

__all__ = [
    "COMPACT_KEEP_LAST",
    "ELIDED_NOTE",
    "ELIDED_SEARCH_NOTE",
    "MAX_RESPONSE",
    "MAX_RESULTS",
    "MAX_SNIPPET",
    "REPORT_FACT",
    "REPORT_SEPARATOR",
    "SCREEN_REMINDER_SEPARATOR",
    "SEARCH_NOT_CONNECTED",
    "LoopStep",
    "action_payload",
    "compact",
    "continuation_chars",
    "elided_payload",
    "place_overview",
    "render_call",
    "render_response",
    "sanitise",
    "screenshot_payload",
    "search_payload",
    "show_current_image",
    "status_payload",
    "step_prompt",
    "swap_newest_image",
    "with_report_fact",
    "with_screen_reminder",
]

MAX_RESULTS: Final = 5
MAX_SNIPPET: Final = 300
#: chars of one rendered response block, frame included
MAX_RESPONSE: Final = 1500
ELLIPSIS: Final = "…"
SEARCH_NOT_CONNECTED: Final = "search is not connected in this session"
#: what a payload that cannot be bounded by cutting its strings is replaced with
TOO_LONG_REASON: Final = "the tool response was too long to show"

#: the stub's note: an elided response says it was omitted (H-T120-3)
ELIDED_NOTE: Final = "earlier result omitted"
#: a search stub's note: the model may search again (H-T120b-0, H-T120b-1); an action's stub does not invite a
#: repeat, since calling an action again would repeat the action
ELIDED_SEARCH_NOTE: Final = "earlier result omitted; search again if you need it"
#: T-0178 B1 (ruling H-T178-3): the fact the report step's continuation ends with. A fixed sentence: no page text, no
#: call, no special-token literal.
REPORT_FACT: Final = (
    "You have used all your actions for this request. Do not call a tool. In one or two sentences, tell the user "
    "what you did, what you found, and where you stopped."
)
#: what separates the last tool response from ``REPORT_FACT``
REPORT_SEPARATOR: Final = "\n\n"
#: T-0201 B2 (H-T201-2): what separates the last tool response from a screen reminder
SCREEN_REMINDER_SEPARATOR: Final = "\n\n"
_STUB_NOTES: Final = (ELIDED_NOTE, ELIDED_SEARCH_NOTE)
#: the newest steps whose responses compaction never elides (H-T120-1)
COMPACT_KEEP_LAST: Final = 2

_Q: Final = '<|"|>'
StatusKind = Literal["refused", "failed", "timeout"]
_STATUSES: Final = ("refused", "failed", "timeout")


@dataclass(frozen=True)
class LoopStep:
    """One executed step: the model's text before the call (step 1: the spoken prefix), the validated call, and
    the payload its result renders from (a builder's mapping, or a plain string)."""

    text: str
    name: str
    arguments: Mapping[str, Any]
    response: Any


# --------------------------------------------------------------------------- the sanitiser


def sanitise(value: Any) -> Any:
    """``value`` with ``drop_special_literals`` applied to every string in it, keys included."""
    if isinstance(value, str):
        return drop_special_literals(value)
    if isinstance(value, Mapping):
        return {sanitise(k): sanitise(v) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [sanitise(v) for v in value]
    return value


# --------------------------------------------------------------------------- the template's rendering


def _sort_key(item: tuple[Any, Any]) -> Any:
    key = item[0]
    return key.lower() if isinstance(key, str) else key


def _entries(mapping: Mapping[Any, Any]) -> str:
    """``key:value,...`` in jinja ``dictsort`` order (case-insensitive, stable), keys unescaped."""
    return ",".join(f"{k}:{_format(v)}" for k, v in sorted(mapping.items(), key=_sort_key))


def _format(value: Any) -> str:
    """The template's ``format_argument(value, escape_keys=False)``."""
    if isinstance(value, str):
        return _Q + value + _Q
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return "{" + _entries(value) + "}"
    if isinstance(value, Sequence):
        return "[" + ",".join(_format(v) for v in value) + "]"
    return str(value)


def _response_block(name: str, payload: Any) -> str:
    if isinstance(payload, Mapping):
        body = "response:" + name + "{" + _entries(payload) + "}"
    else:
        body = "response:" + name + "{value:" + _format(payload) + "}"
    return "<|tool_response>" + body + "<tool_response|>"


def render_call(name: str, arguments: Mapping[str, Any]) -> str:
    """``<|tool_call>call:NAME{...}<tool_call|>`` as the template renders a ``tool_calls`` entry, sanitised.
    Renders what it is given: validating the call is ``live``'s."""
    return (
        "<|tool_call>call:"
        + drop_special_literals(name)
        + "{"
        + _entries(sanitise(arguments))
        + "}<tool_call|>"
    )


# --------------------------------------------------------------------------- bounds


def _cut(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1] + ELLIPSIS if n >= 1 else ""


def _cap(value: Any, n: int) -> Any:
    if isinstance(value, str):
        return _cut(value, n)
    if isinstance(value, Mapping):
        return {k: _cap(v, n) for k, v in value.items()}
    if isinstance(value, Sequence):
        return [_cap(v, n) for v in value]
    return value


def _longest(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return max((_longest(v) for v in value.values()), default=0)
    if isinstance(value, Sequence):
        return max((_longest(v) for v in value), default=0)
    return 0


def render_response(name: str, payload: Any) -> str:
    """``<|tool_response>response:NAME{...}<tool_response|>``: the template's ``format_tool_response_block``
    (a mapping renders its entries, anything else as ``{value:...}``), sanitised and at most ``MAX_RESPONSE``
    chars. Over the bound, every string value is capped at the largest common length that fits."""
    name = drop_special_literals(name)
    clean = sanitise(payload)
    block = _response_block(name, clean)
    if len(block) <= MAX_RESPONSE:
        return block
    lo, hi = -1, _longest(clean)  # the cap that fits is in [lo + 1, hi - 1]; hi is known not to fit
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if len(_response_block(name, _cap(clean, mid))) <= MAX_RESPONSE:
            lo = mid
        else:
            hi = mid
    if lo < 0:  # even with every string empty the structure is too long
        return _response_block(name, status_payload("failed", TOO_LONG_REASON))
    return _response_block(name, _cap(clean, lo))


# --------------------------------------------------------------------------- payload builders (T-0111a §4)


def search_payload(results: SearchResults | None) -> dict[str, Any]:
    """A ``google_search`` response: the query and at most ``MAX_RESULTS`` results (snippets cut to
    ``MAX_SNIPPET``), or, with no backend, a note that search is not connected (H-T111a-9)."""
    if results is None:
        return {"note": SEARCH_NOT_CONNECTED, "results": [], "status": "dry_run"}
    return {
        "query": drop_special_literals(results.query),
        "results": [
            {
                "snippet": _cut(drop_special_literals(r.snippet), MAX_SNIPPET),
                "title": drop_special_literals(r.title),
                "url": drop_special_literals(r.url),
            }
            for r in results.results[:MAX_RESULTS]
        ],
    }


def action_payload(action: str) -> dict[str, Any]:
    """A dry-run action's response: what would have been done; the screen did not change."""
    return {"action": action, "screen_changed": False, "status": "dry_run"}


def status_payload(status: StatusKind, reason: str) -> dict[str, Any]:
    """A refused, failed or timed-out call's response."""
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES}, not {status!r}")
    return {"reason": reason, "status": status}


def screenshot_payload(payload: Mapping[str, Any], updated: bool) -> dict[str, Any]:
    """A live step's response (T-0111g B8): ``payload`` plus ``screenshot``, ``"updated"`` when the next step's
    image was replaced, else ``"unchanged"``. The input is not mutated."""
    out = dict(payload)
    out["screenshot"] = "updated" if updated else "unchanged"
    return out


# --------------------------------------------------------------------------- the live screen (T-0111g B8)


def swap_newest_image(prompt: Prompt, frame: Frame) -> Prompt:
    """``prompt`` with the newest ``ImagePart`` of its final user message replaced by ``frame``'s part, encoded as
    the prompt builder encodes a current frame (``data:image/jpeg;base64,...``, ``phash=frame.phash``). Every other
    message, part and field is unchanged. A final user message with no image (no frame this turn, or one riding on
    the newest history image) gives the prompt back unchanged: a history image is never swapped."""
    users = [i for i, m in enumerate(prompt.messages) if m.role == "user"]
    if not users:
        return prompt
    user = users[-1]
    message = prompt.messages[user]
    slots = [j for j, p in enumerate(message.content) if isinstance(p, ImagePart)]
    if not slots:
        return prompt
    content = list(message.content)
    content[slots[-1]] = ImagePart(
        url="data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode(), phash=frame.phash
    )
    messages = list(prompt.messages)
    messages[user] = Message(role=message.role, content=tuple(content))
    return prompt.model_copy(update={"messages": tuple(messages)})


def show_current_image(prompt: Prompt, frame: Frame) -> Prompt:
    """What a live step request shows (T-0111g B8, the reading for a turn whose frame rides on a history image):
    ``swap_newest_image`` when the final user message holds an image. When it holds none but an earlier message
    does (the turn's frame equalled the newest history image, so the builder did not attach it twice), ``frame``'s
    part is placed first in the final user message, where the builder puts a current frame; the history image is
    never touched. A prompt with no image at all (it says so in words) comes back unchanged."""
    swapped = swap_newest_image(prompt, frame)
    if swapped is not prompt:
        return swapped
    users = [i for i, m in enumerate(prompt.messages) if m.role == "user"]
    has_image = any(isinstance(p, ImagePart) for m in prompt.messages for p in m.content)
    if not users or not has_image:
        return prompt
    user = users[-1]
    message = prompt.messages[user]
    part = ImagePart(
        url="data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode(), phash=frame.phash
    )
    messages = list(prompt.messages)
    messages[user] = Message(role=message.role, content=(part, *message.content))
    return prompt.model_copy(update={"messages": tuple(messages)})


def label_current_image(prompt: Prompt, text: str) -> Prompt:
    """T-0134 A3: ``prompt`` with one ``TextPart(text)`` placed immediately after the newest ``ImagePart`` of its
    final user message (the current frame, wherever ``show_current_image`` put it). A final user message with no
    image (a blind turn, or a frame riding on the newest history image) gives the prompt back unchanged: a history
    image is never labelled. Every other message, part and field is unchanged."""
    users = [i for i, m in enumerate(prompt.messages) if m.role == "user"]
    if not users:
        return prompt
    user = users[-1]
    message = prompt.messages[user]
    slots = [j for j, p in enumerate(message.content) if isinstance(p, ImagePart)]
    if not slots:
        return prompt
    content = list(message.content)
    content.insert(slots[-1] + 1, TextPart(text=text))
    messages = list(prompt.messages)
    messages[user] = Message(role=message.role, content=tuple(content))
    return prompt.model_copy(update={"messages": tuple(messages)})


def place_overview(prompt: Prompt, frame: Frame, text: str) -> Prompt:
    """T-0140 A7: ``prompt`` with ``ImagePart(frame)`` (encoded as the prompt builder encodes a frame) and
    ``TextPart(text)`` inserted immediately before the newest ``ImagePart`` of its final user message (the current
    frame, wherever ``show_current_image`` put it), so the current frame stays the newest image. A final user message
    with no image gives the prompt back unchanged. Every other message, part and field is unchanged."""
    users = [i for i, m in enumerate(prompt.messages) if m.role == "user"]
    if not users:
        return prompt
    user = users[-1]
    message = prompt.messages[user]
    slots = [j for j, p in enumerate(message.content) if isinstance(p, ImagePart)]
    if not slots:
        return prompt
    content = list(message.content)
    part = ImagePart(
        url="data:image/jpeg;base64," + base64.b64encode(frame.jpeg).decode(), phash=frame.phash
    )
    content[slots[-1] : slots[-1]] = [part, TextPart(text=text)]
    messages = list(prompt.messages)
    messages[user] = Message(role=message.role, content=tuple(content))
    return prompt.model_copy(update={"messages": tuple(messages)})


# --------------------------------------------------------------------------- the step prompt (form G)


def _pieces(step: LoopStep) -> tuple[str, str, str]:
    """A step's sanitised text, rendered call and rendered response, as ``step_prompt`` embeds them."""
    return (
        drop_special_literals(step.text),
        render_call(step.name, step.arguments),
        render_response(step.name, step.response),
    )


def _continuation(pieces: Sequence[tuple[str, str, str]]) -> str:
    """The final assistant text: every step's pieces in order, leading whitespace dropped (the template's trim)."""
    return "".join(t + c + r for t, c, r in pieces).lstrip()


def step_prompt(prompt: Prompt, steps: Sequence[LoopStep]) -> Prompt:
    """The next step's request: ``prompt`` (the step-1 prompt as sent) unchanged, plus one final assistant message
    ``text_1 + call_1 + response_1 + ...`` continued with ``continue_final_message=True``. Its leading whitespace
    is dropped, as the template's ``trim`` drops it, so the text is its own template content."""
    if not steps:
        raise ValueError("step_prompt needs at least one executed step")
    if prompt.continue_final_message:
        raise ValueError("step_prompt builds on the step-1 prompt, not on a continuation")
    text = _continuation([_pieces(s) for s in steps])
    fields = {f: getattr(prompt, f) for f in Prompt.model_fields}
    fields["messages"] = (
        *prompt.messages,
        Message(role="assistant", content=(TextPart(text=text),)),
    )
    fields["continue_final_message"] = True
    return Prompt(**fields)


def with_report_fact(prompt: Prompt) -> Prompt:
    """T-0178 B1: the step request ``prompt`` with ``REPORT_SEPARATOR + REPORT_FACT`` appended to its continuation
    (append-only: the continuation's bytes are a prefix of the result, every other field and message unchanged)."""
    if not prompt.continue_final_message or not prompt.messages:
        raise ValueError(
            "with_report_fact needs a step request (a continued final assistant message)"
        )
    last = prompt.messages[-1]
    if last.role != "assistant":
        raise ValueError("with_report_fact needs a continued final assistant message")
    text = last.text() + REPORT_SEPARATOR + REPORT_FACT
    message = Message(role="assistant", content=(TextPart(text=text),))
    return prompt.model_copy(update={"messages": (*prompt.messages[:-1], message)})


def with_screen_reminder(prompt: Prompt, reminder: str) -> Prompt:
    """T-0201 B2: the step request ``prompt`` with ``SCREEN_REMINDER_SEPARATOR + reminder`` appended to its
    continuation (append-only: the continuation's bytes are a prefix of the result and are not interpreted; every
    other field and message unchanged). ``ValueError`` on a prompt that does not continue a final assistant message."""
    if not prompt.continue_final_message or not prompt.messages:
        raise ValueError(
            "with_screen_reminder needs a step request (a continued final assistant message)"
        )
    last = prompt.messages[-1]
    if last.role != "assistant":
        raise ValueError("with_screen_reminder needs a continued final assistant message")
    text = last.text() + SCREEN_REMINDER_SEPARATOR + reminder
    message = Message(role="assistant", content=(TextPart(text=text),))
    return prompt.model_copy(update={"messages": (*prompt.messages[:-1], message)})


# --------------------------------------------------------------------------- compaction (T-0120)


def continuation_chars(steps: Sequence[LoopStep]) -> int:
    """The length of the final assistant text ``step_prompt`` builds from ``steps``."""
    return len(_continuation([_pieces(s) for s in steps]))


def _is_stub(payload: Any) -> bool:
    return isinstance(payload, Mapping) and payload.get("note") in _STUB_NOTES


def elided_payload(original: Any) -> dict[str, Any]:
    """The stub an elided response becomes (H-T120-3, H-T120b-0): ``ELIDED_SEARCH_NOTE`` plus the ``results`` count
    of a mapping with a ``results`` list (a search response), else ``ELIDED_NOTE`` plus the ``status`` of a mapping
    with one, else ``ELIDED_NOTE`` alone."""
    if isinstance(original, Mapping):
        results = original.get("results")
        if isinstance(results, list | tuple):
            return {"note": ELIDED_SEARCH_NOTE, "results": len(results)}
        if "status" in original:
            return {"note": ELIDED_NOTE, "status": original["status"]}
    return {"note": ELIDED_NOTE}


def compact(
    steps: Sequence[LoopStep],
    budget: int,
    *,
    keep_last: int = COMPACT_KEEP_LAST,
    target: int | None = None,
) -> tuple[LoopStep, ...]:
    """``steps`` with older responses elided so the continuation fits. If ``continuation_chars(steps) <= budget``
    the steps come back unchanged (the same objects). Otherwise the responses of ``steps[:-keep_last]`` are
    replaced by ``elided_payload`` stubs one at a time, oldest first, stopping at the first point where the
    continuation is at most ``target`` (default ``budget * 2 // 3``). A stub is never re-elided, and a response
    whose stub does not render shorter is never elided. The result may still be over ``budget``: the caller
    decides."""
    if keep_last < 1:
        raise ValueError(f"keep_last must be at least 1, not {keep_last}")
    if budget < 1:
        raise ValueError(f"budget must be at least 1, not {budget}")
    goal = budget * 2 // 3 if target is None else target
    if goal > budget:
        raise ValueError(f"target {goal} is over the budget {budget}")
    out = list(steps)
    pieces = [_pieces(s) for s in out]
    if len(_continuation(pieces)) <= budget:
        return tuple(out)
    for i in range(len(out) - keep_last):
        step = out[i]
        if _is_stub(step.response):
            continue
        stub = elided_payload(step.response)
        rendered = render_response(step.name, stub)
        if len(rendered) >= len(pieces[i][2]):
            continue
        out[i] = LoopStep(text=step.text, name=step.name, arguments=step.arguments, response=stub)
        pieces[i] = (pieces[i][0], pieces[i][1], rendered)
        if len(_continuation(pieces)) <= goal:
            break
    return tuple(out)
