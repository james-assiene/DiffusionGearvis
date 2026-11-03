"""T-0181 group A pin moves (docs/tasks/T-0181.md, reading A-R13; test-author, A).

Not a test module. One shared normaliser for every existing row that compares the keywords ``cli`` passes to
``Session(...)`` with a BASE (a literal commit, a frozen expectation or a BASE subprocess): T-0181 A3 adds exactly one
keyword, ``mic_unavailable``, and A-R13 says ``cli`` passes it **only** when the mic is unavailable (``--silent``,
``--no-mic``, ``--mic none``; never ``None``), so an available-mic construction is byte-identical to BASE and an
unavailable one differs by that one key.

``without_mic_unavailable`` drops that key and nothing else. It never drops a key whose value BASE also had, never
rewrites a value, and accepts the shapes the rows already hold: one keyword mapping, or a list of them (one per
``Session`` built). Nothing here imports a T-0181 symbol, so the rows that use it stay green at BASE (the key is absent
there, and dropping an absent key is the identity).

The hello pin move (A4, A-R8): every hello reply gains ``astra.mic.state`` directly after ``astra.log.state`` (fourth on
a fresh hello, fifth after a replayed reset), and nothing else in it moves. ``hello_head`` takes a row's BASE live head
(classes or type strings, the log state among them) and returns it with the mic state inserted right after the log
state **when the tree under test defines** ``AstraMicState`` (looked up by name, so BASE, where it does not exist,
keeps its head unchanged). ``MIC_TYPES`` is the pair of message types A1 adds, for the type-set pins (empty at BASE).
A tree that defines the class but leaves the state out of a hello, puts it elsewhere, or sends it twice reddens the
rows that read their head through ``hello_head``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from local_astra.contracts import protocol as _protocol

#: A-R1/A-R13: the one ``Session`` keyword T-0181 A3 adds
MIC_KWARG = "mic_unavailable"

#: A1/A4: ``AstraMicState`` on the tree under test, by name (``None`` at BASE, which has no such class)
MIC_STATE: type | None = getattr(_protocol, "AstraMicState", None)
MIC_STATE_TYPE = "astra.mic.state"
#: A1: the two types T-0181 adds to ``ASTRA_MESSAGE_TYPES`` (client->server mute, server->client state); empty at BASE
MIC_TYPES: frozenset[str] = (
    frozenset({"astra.mic.mute", MIC_STATE_TYPE}) if MIC_STATE is not None else frozenset()
)
MIC_CLIENT_TO_SERVER: frozenset[str] = MIC_TYPES - {MIC_STATE_TYPE}
MIC_SERVER_TO_CLIENT: frozenset[str] = MIC_TYPES & {MIC_STATE_TYPE}


def hello_head(head: Sequence[Any]) -> list[Any]:
    """A row's BASE hello live head (message classes, or type strings) with A4's ``astra.mic.state`` inserted directly
    after the one log state, in the head's own kind; unchanged at BASE. The head must hold exactly one log state."""
    if MIC_STATE is None:
        return list(head)
    at = [i for i, h in enumerate(head) if h is _protocol.AstraLogState or h == "astra.log.state"]
    assert len(at) == 1, f"premise: a BASE hello head holds exactly one log state; got {list(head)}"
    mic: Any = MIC_STATE_TYPE if isinstance(head[at[0]], str) else MIC_STATE
    return [*head[: at[0] + 1], mic, *head[at[0] + 1 :]]


def _type_of(message: Any) -> Any:
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except ValueError:
            return None
    if isinstance(message, Mapping):
        return message.get("type")
    return getattr(message, "type", None)


def hello_without_mic_state(reply: Sequence[Any]) -> list[Any]:
    """A whole hello reply as a row holds it (raw JSON frames, decoded dicts or message objects) with A4's one
    ``astra.mic.state`` taken out, **after asserting** that there is exactly one and that it sits directly after the
    one ``astra.log.state``; unchanged at BASE. What is left is what the row compares with BASE."""
    if MIC_STATE is None:
        return list(reply)
    types = [_type_of(m) for m in reply]
    mic = [i for i, t in enumerate(types) if t == MIC_STATE_TYPE]
    log = [i for i, t in enumerate(types) if t == "astra.log.state"]
    assert len(mic) == 1 and len(log) == 1 and mic[0] == log[0] + 1, (
        "T-0181 A4 (A-R8): a hello reply carries exactly one astra.mic.state, directly after its one "
        f"astra.log.state; the reply is {types}"
    )
    return [m for i, m in enumerate(reply) if i != mic[0]]


def without_mic_unavailable(kwargs: Any) -> Any:
    """``kwargs`` (a keyword mapping, or a list/tuple of them) with the ``mic_unavailable`` key removed; any other
    value is returned unchanged."""
    if isinstance(kwargs, Mapping):
        return {k: v for k, v in kwargs.items() if k != MIC_KWARG}
    if isinstance(kwargs, (list, tuple)):
        return type(kwargs)(without_mic_unavailable(k) for k in kwargs)
    return kwargs


def run_without_mic_unavailable(run: Any) -> Any:
    """A captured ``cli`` run (a mapping holding a ``session_kwargs`` list, as the BASE-subprocess rows record it)
    with ``without_mic_unavailable`` applied to that one entry; every other entry, and any other value, unchanged."""
    if isinstance(run, Mapping) and "session_kwargs" in run:
        return {**run, "session_kwargs": without_mic_unavailable(run["session_kwargs"])}
    return run


def capture_without_mic_unavailable(capture: Any) -> Any:
    """A whole captured section tree (nested mappings and lists, as the T-0137/T-0179 differentials hold it) with
    ``run_without_mic_unavailable`` applied at every mapping that holds a ``session_kwargs`` entry, and nowhere else:
    a ``mic_unavailable`` key anywhere outside a ``Session`` keyword list is kept."""
    if isinstance(capture, Mapping):
        out = {
            k: capture_without_mic_unavailable(v)
            for k, v in capture.items()
            if k != "session_kwargs"
        }
        if "session_kwargs" in capture:
            out["session_kwargs"] = without_mic_unavailable(capture["session_kwargs"])
        return {k: out[k] for k in capture}  # the original key order
    if isinstance(capture, list):
        return [capture_without_mic_unavailable(v) for v in capture]
    return capture
