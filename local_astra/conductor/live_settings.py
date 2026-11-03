"""The settings the HUD panel may change at a turn boundary, and their validation (T-0213 A1-3). Pure.

:class:`LiveSettings` is a frozen snapshot of the v1-changeable values (the catalog rows with ``v1_changeable``),
the user's instructions and a revision. :func:`validate` applies a change set to a snapshot **atomically**: either
every change is valid and a new snapshot comes back, or nothing changes and the refusals say why.

No rule is re-implemented here. Every value and every cross-rule a ``Profile`` enforces at load is checked by
``config.Profile`` itself, and its refusal text is the config module's own ``*_refusal()`` string; the
instructions are checked by ``prompt_builder.validate_instructions``. Only the three settings that have no
``Profile`` key (``screen_reminder``, ``completion_check``) or no strict validator there (``reground``) get a type
check of their own, with the ``type(v) is bool`` strictness of ``config.py``.

No I/O, no clock, no asyncio.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, NamedTuple

from pydantic import ValidationError

from local_astra.conductor.prompt_builder import validate_instructions
from local_astra.config import (
    DEFAULT_ANNOUNCED_CALL_REASK,
    DEFAULT_FRAME_INTERVAL_MS,
    DEFAULT_LIVE_GUIDANCE,
    DEFAULT_MAX_FRAMES_PER_TURN,
    LIVE_DEFAULT_ACTION_OUTCOME,
    LIVE_DEFAULT_AUTO_OVERVIEW,
    LIVE_DEFAULT_FEEDBACK,
    LIVE_DEFAULT_OVERVIEW_FRAME,
    LIVE_DEFAULT_REGROUND,
    LIVE_DEFAULT_REPEAT_GUARD,
    LIVE_DEFAULT_THOUGHT_CALL_RECOVERY,
    LIVE_DEFAULT_VISUAL_TOKENS,
    Profile,
)
from local_astra.settings_catalog import CATALOG, by_key

__all__ = [
    "CHANGEABLE_KEYS",
    "KNOWN_KEYS",
    "PROFILE_KEYS",
    "LiveSettings",
    "Refusal",
    "describe_value",
    "public_key",
    "unknown_key",
    "validate",
]


@dataclass(frozen=True)
class LiveSettings:
    """A snapshot of the panel-changeable settings (T-0213 A1-3). The defaults are the live defaults (the
    ``config`` constants); outside live, a live-only value is ``None`` (unset), as ``cli``'s resolvers give it.
    ``reground_visual_tokens`` is ``None`` unless ReGround is on (``Profile`` refuses it otherwise).
    ``max_frames_per_turn`` and ``frame_interval_ms`` apply in every mode and are never ``None`` (T-0227 A-3)."""

    visual_tokens: int | None = LIVE_DEFAULT_VISUAL_TOKENS
    reground: bool | None = LIVE_DEFAULT_REGROUND
    reground_visual_tokens: int | None = None
    live_feedback: str | None = LIVE_DEFAULT_FEEDBACK
    thought_call_recovery: bool | None = LIVE_DEFAULT_THOUGHT_CALL_RECOVERY
    auto_overview: bool | None = LIVE_DEFAULT_AUTO_OVERVIEW
    overview_frame: bool | None = LIVE_DEFAULT_OVERVIEW_FRAME
    action_outcome: bool | None = LIVE_DEFAULT_ACTION_OUTCOME
    repeat_guard: bool | None = LIVE_DEFAULT_REPEAT_GUARD
    screen_reminder: bool | None = True
    completion_check: bool | None = True
    announced_call_reask: bool | None = DEFAULT_ANNOUNCED_CALL_REASK
    live_guidance: str | None = DEFAULT_LIVE_GUIDANCE
    #: T-0227 A-3: in force in every mode, so never ``None``
    max_frames_per_turn: int = DEFAULT_MAX_FRAMES_PER_TURN
    frame_interval_ms: int = DEFAULT_FRAME_INTERVAL_MS
    user_instructions: str = ""
    rev: int = 0

    def profile_fields(self) -> dict[str, Any]:
        """The snapshot's values that are ``Profile`` fields, by name (``Profile.model_validate`` input)."""
        return {key: getattr(self, key) for key in PROFILE_KEYS}


class Refusal(NamedTuple):
    """One refused change: the key (``"*"`` when a cross-rule names none of the changed keys) and why."""

    key: str
    reason: str


#: Every key a change set may name (the snapshot's fields bar ``rev``).
CHANGEABLE_KEYS: tuple[str, ...] = tuple(f.name for f in fields(LiveSettings) if f.name != "rev")
#: The changeable keys that ``Profile`` validates (its own field validators and cross-rules).
PROFILE_KEYS: tuple[str, ...] = tuple(k for k in CHANGEABLE_KEYS if k in Profile.model_fields)
#: Changeable keys with no ``Profile`` key, live only (``cli`` refuses their flags outside live).
_LIVE_ONLY_FLAG_KEYS: tuple[str, ...] = ("screen_reminder", "completion_check")
#: Keys whose ``Profile`` field has no strict (``type(v) is bool``) validator, or no field at all.
_STRICT_BOOL_KEYS: tuple[str, ...] = ("reground", "screen_reminder", "completion_check")
#: Engine settings (the catalog's ``engine`` row): named so a change to one is refused as a relaunch, not unknown.
_ENGINE_KEYS: frozenset[str] = frozenset(
    {
        "canvas_length",
        "max_denoising_steps",
        "max_num_seqs",
        "kv_cache_memory_bytes",
        "max_model_len",
        "diffusion_config",
        "stt_backend",
        "tts_backend",
        "profile",
    }
)


#: Every key a refusal may name as it is (T-0213 R-19 a/d): the catalog's keys, the changeable keys and the engine
#: settings. Any other key is the page's own text, so it is never repeated back: see :func:`unknown_key`.
KNOWN_KEYS: frozenset[str] = frozenset(
    {e.key for e in CATALOG} | set(CHANGEABLE_KEYS) | _ENGINE_KEYS
)


def unknown_key(key: str) -> str:
    """T-0213 R-19 (a): a key no catalog row names, as a reason states it: its length only, never its text."""
    return f"<unknown key, {len(key)} chars>"


def public_key(key: str) -> str:
    """T-0213 R-19 (d): a refusal's key as it goes on the wire and into the log: a known key as it is, ``"*"`` for
    an empty, unknown or otherwise invalid one (so ``SettingRefusal(min_length=1)`` always holds)."""
    return key if key == "*" or key in KNOWN_KEYS else "*"


def describe_value(value: object) -> str:
    """T-0213 R-19 (a): a value as a reason states it: its type and length, never the value itself (the A1 way,
    as ``validate_instructions`` gives a length and never the text). No ``repr``/``str`` of the value is taken,
    so a huge integer cannot raise here either."""
    if value is None:
        return "<null>"
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, int):
        return "<int>"
    if isinstance(value, float):
        return "<float>"
    if isinstance(value, str):
        return f"<str, {len(value)} chars>"
    if isinstance(value, Mapping):
        return f"<object, {len(value)} keys>"
    if isinstance(value, list | tuple):
        return f"<list, {len(value)} items>"
    return f"<{type(value).__name__}>"


def _key_refusal(key: str) -> str | None:
    """Why ``key`` cannot be changed from the panel, or ``None`` when it can."""
    if key in CHANGEABLE_KEYS:
        return None
    entry = by_key("engine" if key in _ENGINE_KEYS else key)
    if entry is None:
        return f"{unknown_key(key)} is not a setting the panel knows (T-0213)"
    if entry.kind == "fixed":
        return f"{key} is fixed and cannot be changed: {entry.evidence} (T-0213)"
    command = f": relaunch with `{entry.relaunch}`" if entry.relaunch else ""
    if entry.kind == "relaunch":
        return f"{key} changes only at launch{command} (T-0213)"
    return f"{key} cannot be changed from the panel yet (T-0213 D){command}"


def _bool_refusal(key: str, value: object) -> str:
    return f"{key} = {describe_value(value)} is refused: only true and false are allowed (T-0213)"


def _none_refusal(key: str) -> str:
    return f"{key} = None is refused: it is not a value the setting can take (T-0213)"


def _live_only_refusal(key: str, mode: str) -> str:
    return (
        f"{key} with tools = {mode!r} is refused: it applies only with tools = 'live', and elsewhere it would "
        "silently do nothing (T-0213)"
    )


def _without_value(key: str, reason: str, value: object) -> str:
    """T-0213 R-19 (a): ``Profile``'s refusal of ``key`` with the value it quotes replaced by its type and length.
    A per-field refusal of ``config`` reads ``"{key} = {value!r} is refused: <why>"``; the quoted value is cut out
    by position (the text between ``"{key} = "`` and the first ``" is refused"``), so no ``repr`` is compared."""
    head, mark = f"{key} = ", " is refused"
    if reason.startswith(head) and mark in reason:
        return f"{head}{describe_value(value)}{reason[reason.index(mark) :]}"
    return reason


def _profile_refusals(data: Mapping[str, object], changed: tuple[str, ...]) -> list[Refusal]:
    """``Profile``'s own refusals for ``data``, each with the key it is about. A refusal about a changed key never
    quotes the page's value (R-19 a): only its type and length."""
    try:
        Profile.model_validate(dict(data))
    except ValidationError as exc:
        out: list[Refusal] = []
        for err in exc.errors():
            ctx = err.get("ctx") or {}
            reason = str(ctx["error"]) if "error" in ctx else str(err["msg"])
            loc = err.get("loc") or ()
            # a cross-rule has no field: it is about the changed key its text names first
            named = sorted((reason.find(k), k) for k in changed if k in reason)
            key = str(loc[0]) if loc else (named[0][1] if named else "*")
            if (
                loc and key in changed
            ):  # a field refusal quotes the value; a cross-rule only valid ones
                reason = _without_value(key, reason, data[key])
            out.append(Refusal(key, reason))
        return out
    return []


def validate(
    current: LiveSettings,
    changes: Mapping[str, object],
    *,
    mode: str,
    web: bool,
    fixed: Mapping[str, object] | None = None,
) -> tuple[LiveSettings | None, list[Refusal]]:
    """Apply ``changes`` to ``current`` all or nothing (T-0213 A1-3).

    Returns ``(new, [])`` when every change is valid: ``new.rev`` is ``current.rev + 1``, or ``current`` itself
    when nothing changes (an empty or no-op set). Returns ``(None, refusals)`` otherwise, and nothing is changed.
    ``mode`` is the session's resolved tools mode and ``web`` its resolved ``browser_web``; both are fixed for the
    session and take part in the ``Profile`` cross-rules. ``fixed`` holds the other session-fixed ``Profile`` values
    (at least ``page_overview``); they are merged into the same ``Profile`` check, so a change that a fixed value
    rules out (``overview_frame`` or ``auto_overview`` beside a session page overview) is refused with
    ``Profile``'s own text (T-0213 R-6). ``None`` adds nothing."""
    refusals: list[Refusal] = []
    accepted: dict[str, object] = {}
    reground_after = changes.get("reground", current.reground)
    for key, value in changes.items():
        why = _key_refusal(key)
        # ``None`` is ``Profile``'s "unset", not a value to change to; only a ReGround budget may be cleared, and
        # only while ReGround is off (A1-3 strict typing)
        if (
            why is None
            and value is None
            and not (key == "reground_visual_tokens" and reground_after is not True)
        ):
            why = _none_refusal(key)
        if why is None and key in _STRICT_BOOL_KEYS and type(value) is not bool:
            why = _bool_refusal(key, value)
        if why is None and key in _LIVE_ONLY_FLAG_KEYS and mode != "live":
            why = _live_only_refusal(key, mode)
        if why is None and key == "user_instructions":
            if type(value) is not str:
                why = f"user_instructions = {describe_value(value)} is refused: only text is allowed (T-0213)"
            else:
                why = validate_instructions(value)
                value = value.strip()
        if why is not None:
            refusals.append(Refusal(key, why))
        else:
            accepted[key] = value

    changed = tuple(k for k in PROFILE_KEYS if k in accepted)
    # In live every value is in force, so the whole merged snapshot is checked (cross-rules included); elsewhere
    # the live-only values are unset and do nothing, so only the changed ones are checked against the mode.
    merged = current.profile_fields() | {k: accepted[k] for k in changed}
    checked = merged if mode == "live" else {k: merged[k] for k in changed}
    data: dict[str, object] = {
        "name": "lean",
        "tools": mode,
        "browser_web": web,
        **(fixed or {}),
        **checked,
    }
    refusals.extend(_profile_refusals(data, changed))

    if refusals:
        return None, refusals
    if all(
        getattr(current, k) == v and type(getattr(current, k)) is type(v)
        for k, v in accepted.items()
    ):
        return current, []
    updates: dict[str, Any] = dict(accepted)
    return replace(current, **updates, rev=current.rev + 1), []
