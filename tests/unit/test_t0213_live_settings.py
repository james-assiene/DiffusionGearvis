"""T-0213 A1-3: ``conductor.live_settings.LiveSettings`` and ``validate`` reuse the load rules of ``config.Profile``.

The interfaces are the task's: ``LiveSettings`` is a frozen snapshot of the v1-changeable values plus
``user_instructions: str`` and ``rev: int``; ``validate(current, changes, *, mode, web) -> (LiveSettings | None,
refusals)`` applies a change set atomically. Where the task is silent, the choice is fixed here and listed in the
test-author's report:

- ``LiveSettings``'s field names are the catalog keys of the v1-changeable rows (``visual_tokens``, ``reground``,
  ``reground_visual_tokens``, ``live_feedback``, ``thought_call_recovery``, ``auto_overview``, ``overview_frame``,
  ``action_outcome``, ``repeat_guard``, ``screen_reminder``, ``completion_check``, ``announced_call_reask``,
  ``live_guidance``, ``user_instructions``) plus ``rev``, constructed by keyword. ``reground_visual_tokens`` is
  ``int | None`` (``None`` with ReGround off, the only value ``Profile`` accepts there).
- ``refusals`` is a list of ``live_settings.Refusal(key, reason)`` NamedTuples (ruling R-3). Rows check that one
  reason *contains* the config module's own ``*_refusal()`` text, and, where a single key is at fault, that a
  refusal names it.
- ``validate_instructions`` returns ``None`` for accepted text and the refusal reason (a str) otherwise (R-3; see
  ``test_t0213_instructions_block.py``).
- A v1 key's value is never ``None`` in live (``reground_visual_tokens`` with ReGround off excepted): the snapshot
  holds resolved values, and ``None`` is ``Profile``'s "unset", not a value to change to; ``type(v) is bool``
  refuses it (A1-3's strictness).
- R-4: turning ReGround on from the defaults (overview frame on) needs ``overview_frame: False`` in the same set.

Oracles: ``config.Profile.model_validate`` with ``tools = "live"`` is the independent judge of a snapshot (the task's
property), and the ``*_refusal()`` functions are the expected texts. Nothing of the rules is re-implemented here:
the brute-force property accepts a change set iff ``Profile`` accepts the merged values (and the three keys that are
not ``Profile`` fields, ``screen_reminder``, ``completion_check``, ``user_instructions``, are well typed).

The rows were written ahead of A1 and activated at A1 (implementation d7a5e87d): at BASE (82ee4503) the module does
not exist, so each fails on a named ``AssertionError`` ("... does not exist yet").

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pydantic import ValidationError

from local_astra import config
from local_astra.config import Profile

REPO = Path(__file__).resolve().parents[2]

#: A valid live state: the live defaults (config.py), ReGround off so its pass-2 budget is unset.
LIVE: dict[str, Any] = {
    "visual_tokens": 1120,
    "reground": False,
    "reground_visual_tokens": None,
    "live_feedback": "label",
    "thought_call_recovery": True,
    "auto_overview": False,
    "overview_frame": True,
    "action_outcome": False,
    "repeat_guard": True,
    "screen_reminder": True,
    "completion_check": True,
    "announced_call_reask": True,
    "live_guidance": "v3",
    "user_instructions": "",
    "rev": 0,
}
NOT_PROFILE = ("screen_reminder", "completion_check", "user_instructions", "rev")


def _mod() -> Any:
    try:
        return importlib.import_module("local_astra.conductor.live_settings")
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "T-0213 A1-3: local_astra/conductor/live_settings.py does not exist yet"
        ) from exc


def _state(**over: Any) -> Any:
    mod = _mod()
    assert hasattr(mod, "LiveSettings"), (
        "T-0213 A1-3: live_settings.LiveSettings does not exist yet"
    )
    return mod.LiveSettings(**{**LIVE, **over})


_UNSET: Any = object()


def _validate(
    current: Any,
    changes: Mapping[str, Any],
    *,
    mode: str = "live",
    web: bool = True,
    fixed: Any = _UNSET,
) -> Any:
    """``fixed`` is passed only when a row names it (R-6), so the other rows do not depend on the keyword."""
    mod = _mod()
    assert hasattr(mod, "validate"), "T-0213 A1-3: live_settings.validate does not exist yet"
    kw: dict[str, Any] = {"mode": mode, "web": web}
    if fixed is not _UNSET:
        params = inspect.signature(mod.validate).parameters
        assert "fixed" in params and params["fixed"].default is None, (
            f"T-0213 R-6: validate(..., fixed=None) does not exist yet; it takes {list(params)}"
        )
        kw["fixed"] = fixed
    result = mod.validate(current, dict(changes), **kw)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"validate returns (LiveSettings | None, refusals): {result!r}"
    )
    return result


def _reasons(refusals: Iterable[Any]) -> list[str]:
    """The reasons of R-3's ``Refusal(key, reason)`` NamedTuples (a named failure for any other shape)."""
    refusal = getattr(_mod(), "Refusal", None)
    assert refusal is not None, "T-0213 A1-3 (R-3): live_settings.Refusal does not exist"
    out: list[str] = []
    for r in refusals:
        assert isinstance(r, refusal), f"R-3: a refusal is a Refusal(key, reason); got {r!r}"
        assert r._fields == ("key", "reason"), (
            f"R-3: Refusal's fields are (key, reason): {r._fields}"
        )
        assert isinstance(r.key, str) and isinstance(r.reason, str) and r.reason.strip(), repr(r)
        out.append(r.reason)
    return out


def _refused(
    current: Any,
    changes: Mapping[str, Any],
    *,
    want: str | None = None,
    key: str | None = None,
    **kw: Any,
) -> list[str]:
    snap, refusals = _validate(current, changes, **kw)
    reasons = _reasons(refusals)
    assert snap is None, f"{changes!r} must be refused; got {snap!r}"
    assert reasons, f"{changes!r}: a refusal carries a reason"
    if want is not None:
        assert any(want in r for r in reasons), (
            f"{changes!r}: no reason contains config's text {want!r}: {reasons}"
        )
    if key is not None:
        assert key in {r.key for r in refusals}, (
            f"{changes!r}: no refusal names {key!r}: {list(refusals)}"
        )
    return reasons


def _accepted(current: Any, changes: Mapping[str, Any], **kw: Any) -> Any:
    snap, refusals = _validate(current, changes, **kw)
    assert snap is not None, f"{changes!r} must be accepted; refused: {_reasons(refusals)}"
    assert not list(refusals), f"an accepted set has no refusals: {refusals!r}"
    return snap


def _as_profile(snap: Any) -> Profile:
    values = {k: getattr(snap, k) for k in LIVE if k not in NOT_PROFILE}
    return Profile.model_validate({"name": "lean", "tools": "live", **values})


# --- the shape ------------------------------------------------------------------------------------------------


def test_snapshot_is_frozen_and_holds_the_fields() -> None:
    s = _state()
    for k, v in LIVE.items():
        assert getattr(s, k) == v, f"LiveSettings.{k}"
    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass or frozen pydantic model
        s.visual_tokens = 560  # type: ignore[misc]
    _as_profile(s)  # premise: the base state is a valid live profile


def test_valid_set_bumps_rev_and_keeps_current() -> None:
    s = _state(rev=4)
    new = _accepted(s, {"visual_tokens": 560, "action_outcome": True})
    assert new.rev == 5 and new.visual_tokens == 560 and new.action_outcome is True
    assert new.live_guidance == "v3" and new.repeat_guard is True, (
        "untouched keys keep their values"
    )
    assert s.visual_tokens == 1120 and s.rev == 4, "the current snapshot is unchanged"


def test_empty_and_no_op_sets_keep_the_rev() -> None:
    s = _state(rev=3)
    for changes in ({}, {"visual_tokens": 1120}, {"live_guidance": "v3", "repeat_guard": True}):
        new = _accepted(s, changes)
        assert new.rev == 3, f"{changes!r} changes nothing, so the rev stays 3; got {new.rev}"
        assert all(getattr(new, k) == getattr(s, k) for k in LIVE), f"{changes!r}"


# --- atomic refusal and the cross-rules -------------------------------------------------------------------------


def test_atomic_refusal_action_outcome_without_feedback() -> None:
    """The task's row: from a valid state, ``{action_outcome: True, live_feedback: "off"}`` is refused as a whole,
    with config's own text; neither half is applied."""
    s = _state()
    want = config.action_outcome_without_feedback_refusal()
    _refused(s, {"action_outcome": True, "live_feedback": "off"}, want=want)
    _refused(s, {"live_feedback": "off", "action_outcome": True}, want=want)
    # one half against the other's current value
    _refused(_state(action_outcome=True), {"live_feedback": "off"}, want=want)
    _refused(_state(live_feedback="off"), {"action_outcome": True}, want=want)


def test_cross_rule_is_not_too_wide() -> None:
    """The guard's other direction: each half alone, and the outcome with the other label, are accepted."""
    s = _state()
    assert _accepted(s, {"action_outcome": True}).action_outcome is True
    assert _accepted(s, {"live_feedback": "off"}).live_feedback == "off"
    new = _accepted(s, {"action_outcome": True, "live_feedback": "label+extent"})
    assert (new.action_outcome, new.live_feedback) == (True, "label+extent")
    assert (
        _accepted(
            _state(action_outcome=True), {"action_outcome": False, "live_feedback": "off"}
        ).rev
        == 1
    )


def test_atomic_refusal_applies_no_valid_part() -> None:
    s = _state()
    _refused(s, {"visual_tokens": 560, "thinking": False}, key="thinking")
    _refused(s, {"visual_tokens": 560, "repeat_guard": "yes"}, key="repeat_guard")


def test_r4_reground_on_from_the_defaults() -> None:
    """R-4: from the live defaults (``LiveSettings()``, the overview frame on), ``{reground: True}`` alone is
    refused with Profile's own text; with ``overview_frame: False`` in the same set it is accepted and both apply."""
    mod = _mod()
    defaults = mod.LiveSettings()
    assert defaults.overview_frame is config.LIVE_DEFAULT_OVERVIEW_FRAME is True, (
        "premise: the frame is on"
    )
    assert defaults.reground is config.LIVE_DEFAULT_REGROUND is False, "premise: ReGround is off"
    want = config.overview_frame_with_refusal("reground")
    _refused(defaults, {"reground": True}, want=want)
    new = _accepted(defaults, {"reground": True, "overview_frame": False})
    assert (new.reground, new.overview_frame, new.rev) == (True, False, defaults.rev + 1)
    _as_profile(new)


def test_overview_frame_with_reground_is_refused() -> None:
    want = config.overview_frame_with_refusal("reground")
    _refused(_state(overview_frame=False), {"overview_frame": True, "reground": True}, want=want)
    _refused(_state(), {"reground": True}, want=want)  # overview_frame is on in the current state
    _refused(_state(overview_frame=False, reground=True), {"overview_frame": True}, want=want)
    # not too wide: ReGround with the frame off, and the frame with ReGround off
    assert _accepted(_state(), {"overview_frame": False, "reground": True}).reground is True
    assert _accepted(_state(overview_frame=False), {"overview_frame": True}).overview_frame is True


def test_overview_frame_with_auto_overview_is_refused() -> None:
    want = config.overview_frame_with_refusal("auto_overview")
    _refused(_state(), {"auto_overview": True}, want=want)
    assert (
        _accepted(_state(), {"auto_overview": True, "overview_frame": False}).auto_overview is True
    )


def test_reground_visual_tokens_with_reground_off_is_refused() -> None:
    want = config.reground_visual_tokens_unused_refusal("live", False)
    _refused(_state(), {"reground_visual_tokens": 280}, want=want)
    _refused(
        _state(overview_frame=False, reground=True, reground_visual_tokens=280),
        {"reground": False},
        want=want,
    )
    new = _accepted(_state(overview_frame=False, reground=True), {"reground_visual_tokens": 1120})
    assert new.reground_visual_tokens == 1120


def test_live_only_keys_are_refused_outside_live() -> None:
    """A live-only setting outside live would silently do nothing (``.claude/rules/conductor.md:15``)."""
    reasons = _refused(_state(), {"action_outcome": True}, mode="dry_run", web=False)
    assert any("tools = 'dry_run'" in r for r in reasons), reasons


# --- strict typing ----------------------------------------------------------------------------------------------


#: R-22 (re-pin of the A1 rows): under R-19 (a) a reason never repr()s the value, so a row checks config's own text
#: from ``" is refused"`` on, preceded by ``"{key} = "`` and the value's descriptor. The descriptors are written here
#: by hand from docs/02 §5 (A2-apply delta: ``<str, 4 chars>``, ``<int>``); ``<float>`` and ``<bool>`` follow the
#: same "type, and a length where it has one" rule (R-19 a), and are this module's reading of it.
_IS_REFUSED = " is refused"


def _config_tail(refusal: Any, value: Any) -> str:
    """Config's text from ``" is refused"`` on (independent of how the value is quoted before it)."""
    text = refusal(value)
    assert _IS_REFUSED in text, f"premise: config's text has {_IS_REFUSED!r}: {text!r}"
    return text[text.index(_IS_REFUSED) :]


@pytest.mark.parametrize(
    ("key", "value", "refusal", "desc"),
    [
        ("action_outcome", "true", config.action_outcome_refusal, "<str, 4 chars>"),
        ("action_outcome", 1, config.action_outcome_refusal, "<int>"),
        ("thought_call_recovery", 0, config.thought_call_recovery_refusal, "<int>"),
        ("overview_frame", "on", config.overview_frame_refusal, "<str, 2 chars>"),
        ("auto_overview", None, None, None),
        ("visual_tokens", 1120.0, config.visual_tokens_refusal, "<float>"),
        ("visual_tokens", True, config.visual_tokens_refusal, "<bool>"),
        ("visual_tokens", "1120", config.visual_tokens_refusal, "<str, 4 chars>"),
        ("visual_tokens", 1000, config.visual_tokens_refusal, "<int>"),
        ("visual_tokens", None, None, None),
        ("live_guidance", 3, config.live_guidance_refusal, "<int>"),
        ("live_guidance", "v9", config.live_guidance_refusal, "<str, 2 chars>"),
        ("live_feedback", True, config.live_feedback_refusal, "<bool>"),
        ("announced_call_reask", "off", config.announced_call_reask_refusal, "<str, 3 chars>"),
        ("reground", "true", None, None),
        ("reground", 1, None, None),
        ("reground", 0, None, None),
        ("reground", "on", None, None),
        ("screen_reminder", "on", None, None),
        ("screen_reminder", 1, None, None),
        ("completion_check", None, None, None),
        ("user_instructions", 5, None, None),
        ("user_instructions", None, None, None),
    ],
)
def test_type_strictness(key: str, value: Any, refusal: Any, desc: str | None) -> None:
    want = f"{key} = {desc}{_config_tail(refusal, value)}" if refusal is not None else None
    # ``Profile.reground`` is lax ("true" and 1 load as True), so from the defaults (the overview frame on) a
    # coerced True would be refused by the frame cross-rule instead; with the frame off only strictness refuses it
    state = _state(overview_frame=False) if key == "reground" else _state()
    reasons = _refused(state, {key: value}, want=want, key=key)
    if key == "reground":
        assert not any("overview_frame" in r for r in reasons), (
            f"reground = {value!r} is refused for its type, not by a cross-rule: {reasons}"
        )
    if refusal is not None:
        # R-19 (a): the value's repr is absent. Short reprs ("0", "3") also occur in config's own prose (task ids,
        # 'v3'), so the whole reason is checked only for reprs longer than that; the part naming the value (before
        # " is refused") is checked always.
        rep = repr(value)
        for r in reasons:
            head = r.split(_IS_REFUSED, 1)[0]
            assert rep not in head, f"R-19 (a): the reason quotes the value {rep}: {r!r}"
            if len(rep) > 2:
                assert rep not in r, f"R-19 (a): the reason quotes the value {rep}: {r!r}"


# --- relaunch, fixed, later and unknown keys ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("browser_web", False),
        ("canvas_length", 512),
        ("thinking", False),
        ("tools", "dry_run"),
        ("search", "off"),
        ("action_echo", "pixels"),
        ("page_overview", True),
        ("wait_tool", False),
        ("max_tool_steps", 3),
        ("filler_clips", True),
        ("rev", 9),
        ("zz_not_a_setting", True),
    ],
)
def test_relaunch_fixed_later_and_unknown_keys_are_refused(key: str, value: Any) -> None:
    s = _state()
    _refused(s, {key: value}, key=key)
    _refused(s, {"visual_tokens": 560, key: value}, key=key)  # and atomically


# --- review rows (A1 review at 5718460a): R-6, the ReGround clear, modes, refusal texts ---------------------------


@pytest.mark.parametrize(
    ("changes", "want"),
    [
        ({"overview_frame": True}, config.overview_frame_with_refusal("page_overview")),
        ({"auto_overview": True}, config.auto_and_page_overview_refusal()),
    ],
    ids=["overview_frame", "auto_overview"],
)
def test_r6_fixed_page_overview_takes_part_in_the_cross_rules(
    changes: dict[str, Any], want: str
) -> None:
    """R-6: with the session's ``page_overview`` on (``fixed``), the overview frame and the automatic overview are
    refused with ``Profile``'s own text. The state is the one such a session runs in: the frame off (``Profile``
    refuses it beside the tool)."""
    s = _state(overview_frame=False)
    _refused(s, changes, want=want, fixed={"page_overview": True})
    # the companion: page_overview off in ``fixed`` refuses neither
    assert _accepted(s, changes, fixed={"page_overview": False}).rev == 1


@pytest.mark.parametrize("changes", [{"overview_frame": True}, {"auto_overview": True}])
def test_r6_fixed_none_is_unchanged(changes: dict[str, Any]) -> None:
    """R-6's other direction: ``fixed=None`` is the A1 behaviour, so the same sets are accepted, and a set
    ``Profile`` refuses on its own is still refused with ``fixed=None``."""
    s = _state(overview_frame=False)
    new = _accepted(s, changes, fixed=None)
    assert all(getattr(new, k) == v for k, v in changes.items()) and new.rev == 1
    _refused(
        _state(),
        {"auto_overview": True},
        want=config.overview_frame_with_refusal("auto_overview"),
        fixed=None,
    )


def test_r6_current_must_be_built_as_cli_resolves_it() -> None:
    """R-6, the implementer's note (pins A2's construction of ``current``): with the session's ``page_overview`` on,
    cli resolves the overview frame off (``Profile`` refuses the frame beside the tool), so a ``current`` built that
    way accepts an empty change set unchanged. The bare ``LiveSettings()`` default (the frame on) is not that state:
    with the same ``fixed`` it is refused with ``Profile``'s own text even though nothing changes. A2 must build
    ``current`` from the resolved values, not from the defaults."""
    mod = _mod()
    fixed = {"page_overview": True}
    resolved = _state(overview_frame=False)
    new = _accepted(resolved, {}, fixed=fixed)
    # nothing changes but (perhaps) the revision; the rev of an empty set is not this row's to pin
    assert dataclasses.replace(new, rev=resolved.rev) == resolved, (new, resolved)
    assert new.overview_frame is False
    defaults = mod.LiveSettings()
    assert defaults.overview_frame is True, (
        "premise: the LiveSettings() default has the overview frame on"
    )
    _refused(defaults, {}, want=config.overview_frame_with_refusal("page_overview"), fixed=fixed)


def test_reground_off_clears_its_budget_in_one_set() -> None:
    """M5: from a ReGround-on state with a crop budget, ``{reground: False, reground_visual_tokens: None}`` is
    accepted (the only way off, since ``{reground: False}`` alone is refused), and both apply."""
    s = _state(overview_frame=False, reground=True, reground_visual_tokens=280)
    _as_profile(s)  # premise: a valid live state
    new = _accepted(s, {"reground": False, "reground_visual_tokens": None})
    assert (new.reground, new.reground_visual_tokens, new.rev) == (False, None, s.rev + 1)
    _as_profile(new)


@pytest.mark.parametrize("mode", ["dry_run", "off"])
@pytest.mark.parametrize("key", ["screen_reminder", "completion_check"])
@pytest.mark.parametrize("value", [False, True])
def test_live_only_flags_without_a_profile_key_are_refused_outside_live(
    mode: str, key: str, value: bool
) -> None:
    """M3: the two live-only keys that ``Profile`` does not hold are refused outside live (a silent no-op,
    ``.claude/rules/conductor.md:15``), whatever the value, and the reason names the mode as config's
    ``*_without_live_refusal`` texts do (there is no config text for these two)."""
    reasons = _refused(_state(), {key: value}, key=key, mode=mode, web=False)
    assert any(f"tools = {mode!r}" in r for r in reasons), reasons


@pytest.mark.parametrize("mode", ["dry_run", "off"])
def test_user_instructions_apply_in_every_mode(mode: str) -> None:
    """M4: the instructions block is in every mode's system message (A1-4), so the change is accepted off live."""
    new = _accepted(_state(), {"user_instructions": "Answer in French."}, mode=mode, web=False)
    assert new.user_instructions == "Answer in French." and new.rev == 1


def test_announced_call_reask_follows_profile_outside_live() -> None:
    """M4: ``announced_call_reask`` applies in ``dry_run`` (accepted) and is refused with ``tools = "off"`` with
    config's own text (T-0172 B9)."""
    new = _accepted(_state(), {"announced_call_reask": False}, mode="dry_run", web=False)
    assert new.announced_call_reask is False and new.rev == 1
    _refused(
        _state(),
        {"announced_call_reask": False},
        want=config.announced_call_reask_without_tools_refusal("off"),
        key="announced_call_reask",
        mode="off",
        web=False,
    )


def _catalog_row(key: str) -> Any:
    from local_astra import settings_catalog

    row = settings_catalog.by_key(key)
    assert row is not None, f"premise: the catalog has a {key!r} row"
    return row


@pytest.mark.parametrize(
    "key", ["browser_web", "tools", "search", "max_tool_steps", "filler_clips"]
)
def test_relaunch_refusal_shows_the_rows_command(key: str) -> None:
    """M15: a relaunch key's refusal carries the catalog row's own relaunch command (what B-2 shows the user), and
    does not read as a fixed setting."""
    row = _catalog_row(key)
    assert row.kind == "relaunch" and row.relaunch, (
        f"premise: {key} is a relaunch row with a command"
    )
    reasons = _refused(_state(), {key: False}, key=key)
    assert any(row.relaunch in r for r in reasons), (row.relaunch, reasons)
    assert not any("fixed" in r for r in reasons), reasons


@pytest.mark.parametrize("key", ["canvas_length", "max_denoising_steps", "stt_backend"])
def test_engine_key_refusal_shows_the_engine_command(key: str) -> None:
    """M16: an engine setting is not an unknown key: its refusal carries the engine row's relaunch command."""
    row = _catalog_row("engine")
    assert row.relaunch, "premise: the engine row has a relaunch command"
    reasons = _refused(_state(), {key: 1}, key=key)
    assert any(row.relaunch in r for r in reasons), (row.relaunch, reasons)


def test_fixed_refusal_offers_no_relaunch() -> None:
    """M16: ``thinking`` is fixed (A2a-6): its refusal says so and offers no relaunch command, since none changes
    it; and the three refusal shapes (fixed, relaunch, unknown) are distinct texts."""
    fixed = _refused(_state(), {"thinking": False}, key="thinking")
    assert any("fixed" in r for r in fixed), fixed
    assert not any("relaunch" in r or "diffusion_gearvis" in r for r in fixed), fixed
    relaunch = _refused(_state(), {"browser_web": False}, key="browser_web")
    unknown = _refused(_state(), {"zz_not_a_setting": True}, key="zz_not_a_setting")
    assert not any("diffusion_gearvis" in r for r in unknown), (
        f"an unknown key offers no command: {unknown}"
    )
    assert len({fixed[0], relaunch[0], unknown[0]}) == 3


# --- the instructions -------------------------------------------------------------------------------------------


def test_user_instructions_go_through_validate_instructions() -> None:
    from local_astra.conductor import prompt_builder as pb

    check = getattr(pb, "validate_instructions", None)
    assert callable(check), "T-0213 A1-4: prompt_builder.validate_instructions does not exist yet"
    for bad in ("x" * 1001, "say <|think|>", "left" + chr(0x202E) + "right"):
        reason = check(bad)
        assert isinstance(reason, str) and reason, (
            f"premise: {bad[:20]!r} is refused by validate_instructions"
        )
        _refused(_state(), {"user_instructions": bad}, want=reason, key="user_instructions")
    for bad in (
        "stop at <eos>",
        "a" + chr(0x200D) + "b",
    ):  # R-1 bar-less special, R-2 other Cf (ZWJ)
        _refused(_state(), {"user_instructions": bad}, key="user_instructions")
    new = _accepted(_state(), {"user_instructions": "Answer in French."})
    assert new.user_instructions == "Answer in French." and new.rev == 1
    assert _accepted(new, {"user_instructions": ""}).user_instructions == ""


def test_user_instructions_with_a_tab_are_accepted() -> None:
    """R-2: tab is allowed, so instructions holding one are a valid change."""
    text = "Units:" + chr(0x09) + "metric"
    new = _accepted(_state(), {"user_instructions": text})
    assert new.user_instructions == text and new.rev == 1


# --- no I/O, no clock --------------------------------------------------------------------------------------------


def test_module_is_pure() -> None:
    _mod()
    src = (REPO / "local_astra" / "conductor" / "live_settings.py").read_text(encoding="utf-8")
    banned = {
        "asyncio",
        "time",
        "datetime",
        "socket",
        "os",
        "pathlib",
        "subprocess",
        "httpx",
        "random",
        "threading",
    }
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods = [(node.module or "").split(".")[0]]
        else:
            mods = []
        hit = sorted(set(mods) & banned)
        assert not hit, f"A1-3: live_settings imports {hit} (no I/O, no clock)"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("open", "print", "input"), (
                f"A1-3: live_settings calls {node.func.id}()"
            )


# --- the property ------------------------------------------------------------------------------------------------

_TYPED: dict[str, st.SearchStrategy[Any]] = {
    "visual_tokens": st.sampled_from(config.VISUAL_TOKENS),
    "reground": st.booleans(),
    "live_feedback": st.sampled_from(config.LIVE_FEEDBACK_CHOICES),
    "thought_call_recovery": st.booleans(),
    "auto_overview": st.booleans(),
    "overview_frame": st.booleans(),
    "action_outcome": st.booleans(),
    "repeat_guard": st.booleans(),
    "screen_reminder": st.booleans(),
    "completion_check": st.booleans(),
    "announced_call_reask": st.booleans(),
    "live_guidance": st.sampled_from(config.LIVE_GUIDANCE_VERSIONS),
}
_JUNK = st.sampled_from(["true", "on", 1, 0, 1120.0, "1120", None, "v9", 3])


def _mostly(typed: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    """A typed value seven times in eight, junk otherwise: junk in a set refuses it whole, so an even mix would
    leave few accepted sets to check."""
    return st.integers(0, 7).flatmap(lambda i: _JUNK if i == 0 else typed)


_VALUE = {k: _mostly(v) for k, v in _TYPED.items()}
_CHANGES = st.dictionaries(st.sampled_from(sorted(_TYPED)), st.none(), max_size=4).flatmap(
    lambda d: st.fixed_dictionaries({k: _VALUE[k] for k in d})
)


def _oracle_accepts(current: Any, changes: Mapping[str, Any]) -> bool:
    """Independent judge: the non-Profile keys and ``reground`` are bools, and Profile accepts the merged values in
    live. ``reground`` is listed because ``Profile.reground`` is lax (``0``, ``1`` and ``"on"`` load), while A1-3
    asks for the ``type(v) is bool`` strictness of the other flags."""
    for k in ("screen_reminder", "completion_check", "reground"):
        if k in changes and type(changes[k]) is not bool:
            return False
    if any(v is None for v in changes.values()):
        return False  # no v1 key of this strategy takes None: ``Profile``'s "unset" is not a value to change to
    merged = {k: getattr(current, k) for k in LIVE if k not in NOT_PROFILE}
    merged.update({k: v for k, v in changes.items() if k not in NOT_PROFILE})
    try:
        Profile.model_validate({"name": "lean", "tools": "live", **merged})
    except ValidationError:
        return False
    return True


@given(st.lists(_CHANGES, max_size=6))
def test_validate_accepts_exactly_what_profile_accepts(seq: list[dict[str, Any]]) -> None:
    """Brute force against the load rules: over any sequence of change sets (well typed and junk), ``validate``
    accepts iff ``Profile`` accepts the merged values; an accepted snapshot holds them, validates as a live
    ``Profile``, and its rev moves by one exactly when a value changed. A refused set leaves the state as it was."""
    cur = _state()
    for changes in seq:
        expect = _oracle_accepts(cur, changes)
        snap, refusals = _validate(cur, changes)
        assert (snap is not None) == expect, (
            f"{changes!r} from {cur!r}: oracle {expect}, refusals {_reasons(refusals)}"
        )
        if snap is None:
            continue
        _as_profile(snap)
        for k in LIVE:
            if k != "rev":
                assert getattr(snap, k) == changes.get(k, getattr(cur, k)), f"{k} after {changes!r}"
        changed = any(changes[k] != getattr(cur, k) for k in changes)
        assert snap.rev == cur.rev + (1 if changed else 0), f"rev after {changes!r}"
        cur = snap


_ANY_KEY = dict(
    _VALUE,
    reground_visual_tokens=st.one_of(st.none(), _mostly(st.sampled_from(config.VISUAL_TOKENS))),
)
_ANY_CHANGES = st.dictionaries(st.sampled_from(sorted(_ANY_KEY)), st.none(), max_size=5).flatmap(
    lambda d: st.fixed_dictionaries({k: _ANY_KEY[k] for k in d})
)


@example([{"reground_visual_tokens": 280}])
@example(
    [
        {"overview_frame": False, "reground": True, "reground_visual_tokens": 560},
        {"reground": False},
    ]
)
@example([{"overview_frame": False}, {"reground": True}, {"overview_frame": True}])
@example([{"live_feedback": "off"}, {"action_outcome": True}])
@example([{"auto_overview": True}])
@given(st.lists(_ANY_CHANGES, max_size=8))
def test_every_resulting_snapshot_is_a_valid_live_profile(seq: list[dict[str, Any]]) -> None:
    """The task's property, over every v1 key including the pass-2 budget: whatever is accepted, the snapshot always
    satisfies ``Profile.model_validate`` with ``tools = "live"``, and the rev never goes backwards."""
    cur = _state()
    for changes in seq:
        snap, _ = _validate(cur, changes)
        if snap is None:
            continue
        _as_profile(snap)
        assert snap.rev in (cur.rev, cur.rev + 1)
        cur = snap


# --- T-0227 named re-pin: the two frame keys join the change alphabet (docs/tasks/T-0227.md A-3) ---------------
#
# The two properties above, with ``max_frames_per_turn`` and ``frame_interval_ms`` added to the alphabet: the same
# assertions, the same example budget (the profile's), more keys. Both keys are ``Profile`` fields (A-1), so the
# oracle judges them through ``Profile`` too; their allowed values are typed from A-1, and the junk is aimed at their
# strictness (``True``, a float, a string of an allowed value, out-of-set ints).

T0227_LIVE: dict[str, Any] = {**LIVE, "max_frames_per_turn": 3, "frame_interval_ms": 1000}
_T0227_JUNK = st.sampled_from([True, False, 3.0, "3", "1000", 1000.0, 0, 5, 249, 251, None])


def _t0227_mostly(typed: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    return st.integers(0, 7).flatmap(lambda i: _T0227_JUNK if i == 0 else typed)


_T0227_VALUE = dict(
    _VALUE,
    max_frames_per_turn=_t0227_mostly(st.sampled_from((1, 2, 3, 4))),
    frame_interval_ms=_t0227_mostly(
        st.sampled_from((250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000))
    ),
)
_T0227_CHANGES = st.dictionaries(
    st.sampled_from(sorted(_T0227_VALUE)), st.none(), max_size=4
).flatmap(lambda d: st.fixed_dictionaries({k: _T0227_VALUE[k] for k in d}))
_T0227_ANY_KEY = dict(_T0227_VALUE, reground_visual_tokens=_ANY_KEY["reground_visual_tokens"])
_T0227_ANY_CHANGES = st.dictionaries(
    st.sampled_from(sorted(_T0227_ANY_KEY)), st.none(), max_size=5
).flatmap(lambda d: st.fixed_dictionaries({k: _T0227_ANY_KEY[k] for k in d}))


def _t0227_state() -> Any:
    names = {f.name for f in dataclasses.fields(_mod().LiveSettings)}
    missing = [k for k in ("max_frames_per_turn", "frame_interval_ms") if k not in names]
    assert not missing, f"T-0227 A-3: LiveSettings fields {missing} do not exist yet"
    return _state()


def _t0227_as_profile(snap: Any) -> Profile:
    values = {k: getattr(snap, k) for k in T0227_LIVE if k not in NOT_PROFILE}
    return Profile.model_validate({"name": "lean", "tools": "live", **values})


def _t0227_oracle(current: Any, changes: Mapping[str, Any]) -> bool:
    for k in ("screen_reminder", "completion_check", "reground"):
        if k in changes and type(changes[k]) is not bool:
            return False
    if any(v is None for v in changes.values()):
        return False
    merged = {k: getattr(current, k) for k in T0227_LIVE if k not in NOT_PROFILE}
    merged.update({k: v for k, v in changes.items() if k not in NOT_PROFILE})
    try:
        Profile.model_validate({"name": "lean", "tools": "live", **merged})
    except ValidationError:
        return False
    return True


@example([{"max_frames_per_turn": 4, "frame_interval_ms": 250}])
@example([{"max_frames_per_turn": True}, {"frame_interval_ms": "1000"}])
@given(st.lists(_T0227_CHANGES, max_size=6))
def test_t0227_validate_accepts_exactly_what_profile_accepts(seq: list[dict[str, Any]]) -> None:
    """T-0227 re-pin of ``test_validate_accepts_exactly_what_profile_accepts`` with the two frame keys."""
    cur = _t0227_state()
    for changes in seq:
        expect = _t0227_oracle(cur, changes)
        snap, refusals = _validate(cur, changes)
        assert (snap is not None) == expect, (
            f"{changes!r} from {cur!r}: oracle {expect}, refusals {_reasons(refusals)}"
        )
        if snap is None:
            continue
        _t0227_as_profile(snap)
        for k in T0227_LIVE:
            if k != "rev":
                assert getattr(snap, k) == changes.get(k, getattr(cur, k)), f"{k} after {changes!r}"
        changed = any(changes[k] != getattr(cur, k) for k in changes)
        assert snap.rev == cur.rev + (1 if changed else 0), f"rev after {changes!r}"
        cur = snap


@example([{"max_frames_per_turn": 1}, {"frame_interval_ms": 10000}])
@example([{"overview_frame": False, "reground": True, "max_frames_per_turn": 4}])
@given(st.lists(_T0227_ANY_CHANGES, max_size=8))
def test_t0227_every_resulting_snapshot_is_a_valid_live_profile(seq: list[dict[str, Any]]) -> None:
    """T-0227 re-pin of ``test_every_resulting_snapshot_is_a_valid_live_profile`` with the two frame keys: every
    reachable snapshot validates as a live ``Profile`` (its two frame values included, never ``None``)."""
    cur = _t0227_state()
    for changes in seq:
        snap, _ = _validate(cur, changes)
        if snap is None:
            continue
        _t0227_as_profile(snap)
        assert snap.max_frames_per_turn is not None and snap.frame_interval_ms is not None
        assert snap.rev in (cur.rev, cur.rev + 1)
        cur = snap
