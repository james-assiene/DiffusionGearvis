"""T-0182 group B (amendment B-R1): the one normaliser for rows that compare against BASE.

After B-R5 group B adds nothing to turn lines (B9's ``extras.clips`` moved to T-0182c) and B4's state attribute is
excluded from every ``MachineState`` dump, so a BASE differential can see only these, and nothing else may move:

1. (B9's ``extras.clips`` normaliser was removed with B-R5; T-0182c brings it back if it needs one.)
2. **The new B4 state field** ("no ``still_working`` has played this turn", cleared at turn end and on reset). Its
   name is the implementer's; :data:`BASE_B_STATE_FIELDS` is ``MachineState``'s field list at BASE_B, written out by
   hand from ``git show 7646b8b6:local_astra/contracts/turn.py`` (``test_t0182_differential`` re-derives it from the
   blob). :func:`base_state` keeps exactly those keys of a ``MachineState`` (or of its dump), so a dump taken after B
   compares with one taken at BASE, field for field, on every field BASE had.
3. **Ack-only records** (B1: ``ack`` is off in a session ``cli`` starts with defaults, so such a run no longer plays the
   clip BASE played at every thinking turn's ``llm.start``). After B-R4 ``TurnPolicy()`` keeps BASE's ack, so a rig
   that builds ``TurnPolicy()`` needs none of this; it is for traces of the CLI-default policy (``ack_clip=False``)
   compared with BASE. :func:`without_ack` removes, from any JSON-like tree:

   - every ``play_clip`` command dump whose ``kind`` is ``ack``;
   - every ``stop_clip`` command dump in a ``machine.step``-shaped record whose ``state_before`` is ``THINKING`` and
     that moves to ``SPEAKING``, or whose event is ``session.reset`` (BASE emits those only while ``clip_playing``,
     and with the default policy only the ack sets it before a turn has run 20 s in a live loop);
   - in a machine-section entry (``{"event", "state", "commands"}``, as ``test_session_tools_off_golden`` captures),
     the same two command shapes, read off the entry's own event and the previous entry's state, plus the state's
     ``fillers_this_turn``/``clip_playing``, which only the ack moved;
   - every clip utterance a recorder captured (``{"kind": "ack", "source": ""}``-shaped dicts).

   A ``stop_clip`` on ``user.speech_onset`` in THINKING is **kept**: BASE emits it unconditionally (it is not caused
   by the ack), and F2 keeps it.

   **Validated (2026-10-28, test-author, on group B's prototype)** against the real default and ``ack_clip=True``
   sessions of both golden rigs: on T-0092a's (``test_session_stale_playback_golden``) ``without_ack`` of each side
   is equal in every section (machine, HUD, records, requests). On T-0092b's (``test_session_tools_off_golden``) the
   prompts and requests are equal, and machine, HUD and log are equal up to the TTS-error turn. There the absent ack
   changes the playback run itself, not only its records: without the ack, the failing utterance's run reports a
   ``playback.drained`` (``played_ms`` 0) and an ``astra.spoken.played`` after the turn went IDLE, which the ack-on
   session (and BASE) never emit. That is a session consequence of B1, not an ack-only record, so ``without_ack`` is
   **not** widened to hide it. A session-level BASE differential therefore keeps BASE's ack (``TurnPolicy()``,
   B-R4), and ``without_ack`` is for machine-level traces and for sessions with no TTS error.

The B-R1/B-R3 rig opt-in (``base_equivalent``) is gone with B-R4: a rig that builds ``TurnPolicy()`` is already BASE's.

No assertion is dropped by this module: it only rewrites the observed side (and, where a row says so, the BASE side)
before an equality. A row that uses it says which side it normalises.

Regime: pure; JSON-like trees only.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

#: BASE_B, a literal (``.claude/rules/tests.md``): main when group B branched.
BASE_B = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"

#: ``MachineState``'s fields at BASE_B, in declaration order, by hand from the blob.
BASE_B_STATE_FIELDS: tuple[str, ...] = (
    "state",
    "turn_id",
    "turn_seq",
    "onset_ms",
    "t_eot_ms",
    "words",
    "next_words",
    "turn_phashes",
    "active_generate",
    "active_prefill",
    "committed_answer",
    "spoken_text",
    "held_text",
    "utterances",
    "utterance_seq",
    "timers",
    "history",
    "fillers_this_turn",
    "first_canvas_ms",
    "first_canvas_measured",
    "llm_start_ms",
    "flush_t_ms",
    "pause_onset_ms",
    "played_text",
    "llm_done",
    "tts_done",
    "playback_drained",
    "clip_playing",
    "degraded",
    "tool",
    "tool_repairs",
    "spoken_after_drain",
    "tool_step",
    "tool_steps_executed",
    "awaiting_result",
    "step_text",
    "played_prior",
)

#: ``ClipKind`` at BASE_B and the kinds group B classes as ending (``reflex.clips.ENDING_KINDS``, amendment B-R1).
ENDING_KINDS = frozenset({"out_of_steps", "not_finished", "lost_thread"})


def base_state(state: Any) -> dict[str, Any]:
    """``state`` (a ``MachineState`` or a dump of one) as a dict holding exactly BASE_B's fields that it has."""
    if hasattr(state, "model_dump"):
        state = state.model_dump(mode="json")
    if not isinstance(state, Mapping):
        raise TypeError(f"base_state takes a MachineState or a mapping, not {type(state).__name__}")
    return {k: state[k] for k in BASE_B_STATE_FIELDS if k in state}


def _is_ack_play(node: Any) -> bool:
    return (
        isinstance(node, Mapping) and node.get("type") == "play_clip" and node.get("kind") == "ack"
    )


def _is_stop(node: Any) -> bool:
    return isinstance(node, Mapping) and node.get("type") == "stop_clip"


def _event_type(node: Mapping[str, Any]) -> str | None:
    event = node.get("event")
    return event.get("type") if isinstance(event, Mapping) else None


def _strip_step_record(data: dict[str, Any]) -> None:
    """A ``machine.step`` log record's data: ``event``, ``state_before``, ``state_after``, ``commands``."""
    commands = data.get("commands")
    if not isinstance(commands, list):
        return
    thinking_to_speaking = (
        data.get("state_before") == "THINKING" and data.get("state_after") == "SPEAKING"
    )
    reset = _event_type(data) == "session.reset"
    data["commands"] = [
        c
        for c in commands
        if not _is_ack_play(c) and not (_is_stop(c) and (thinking_to_speaking or reset))
    ]


def _strip_machine_section(entries: list[Any]) -> list[Any]:
    """Entries ``{"event", "state", "commands"}`` in step order (the golden rigs' ``machine`` section)."""
    out: list[Any] = []
    before: str | None = "IDLE"
    for entry in entries:
        if not (isinstance(entry, dict) and "state" in entry and "commands" in entry):
            out.append(entry)
            continue
        state = entry["state"]
        after = state.get("state") if isinstance(state, Mapping) else None
        thinking_to_speaking = before == "THINKING" and after == "SPEAKING"
        reset = _event_type(entry) == "session.reset"
        new = dict(entry)
        new["commands"] = [
            c
            for c in entry["commands"]
            if not _is_ack_play(c) and not (_is_stop(c) and (thinking_to_speaking or reset))
        ]
        if isinstance(state, Mapping):
            new_state = dict(state)
            new_state.pop("fillers_this_turn", None)
            new_state.pop("clip_playing", None)
            new["state"] = new_state
        out.append(new)
        before = after
    return out


def without_ack(tree: Any) -> Any:
    """A deep copy of ``tree`` with the ack-only records removed (module docstring, item 3)."""
    out = copy.deepcopy(tree)

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            if node and all(isinstance(e, dict) and "commands" in e and "state" in e for e in node):
                node = _strip_machine_section(node)
            return [walk(v) for v in node if not _is_ack_play(v) and not _is_ack_clip_play(v)]
        if isinstance(node, dict):
            if node.get("kind") == "machine.step" and isinstance(node.get("data"), dict):
                _strip_step_record(node["data"])
            elif {"state_before", "state_after", "commands"} <= set(node):
                _strip_step_record(node)
            return {k: walk(v) for k, v in node.items()}
        return node

    return walk(out)


def _is_ack_clip_play(node: Any) -> bool:
    """A recorder's clip play: ``{"kind": "ack", "source": ""}`` (no ``type``: not a command dump)."""
    return (
        isinstance(node, Mapping)
        and "type" not in node
        and node.get("kind") == "ack"
        and node.get("source") == ""
    )


__all__ = [
    "BASE_B",
    "BASE_B_STATE_FIELDS",
    "ENDING_KINDS",
    "base_state",
    "without_ack",
]
