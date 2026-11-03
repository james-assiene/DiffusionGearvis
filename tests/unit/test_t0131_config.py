"""T-0131 A1: ``Profile.visual_tokens`` and ``Profile.reground``, and their two load-time refusals
(docs/tasks/T-0131.md, group A).

Expected values are written out from the criterion:

- ``visual_tokens: Literal[280, 560, 1120] | None = None``; ``None`` means "send nothing" (the engine's 280);
- any other value is refused at load, and the refusal names the three allowed values (70, 140, 0, the string
  ``"1120"`` and 1121 are the criterion's and the task's junk list);
- ``reground: bool = False``; ``reground = true`` with ``tools`` other than ``live`` is refused at load, naming
  ``live`` (never a silent no-op, ``.claude/rules/conductor.md``). Its too-wide companion: ``live`` with
  ``reground = true`` loads.

Every row is ``pending(task="T-0131")``: at BASE (f30365d) neither field exists, so each row fails on a named
``AssertionError`` (an ``extra_forbidden`` refusal is turned into one), never on an import or a ``TypeError``.

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from local_astra.config import Profile, load

ALLOWED = (280, 560, 1120)
NEW_FIELDS = ("visual_tokens", "reground")


def _require_fields() -> None:
    missing = [f for f in NEW_FIELDS if f not in Profile.model_fields]
    assert not missing, f"T-0131 A1: Profile has no field(s) {missing} yet"


def _profile(**kw: Any) -> Profile:
    """``Profile(name="lean", **kw)``; a refusal is re-raised as an ``AssertionError`` naming the input."""
    _require_fields()
    try:
        return Profile(name="lean", **kw)
    except ValidationError as exc:
        raise AssertionError(
            f"T-0131 A1: Profile(**{kw!r}) is accepted; refused with {exc}"
        ) from None


def _toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "astra.toml"
    p.write_text('profile = "lean"\n[profiles.lean]\n' + body)
    return p


def _names_allowed(message: str) -> bool:
    return all(str(v) in message for v in ALLOWED)


def test_visual_tokens_values(tmp_path: Path) -> None:
    """A1: 280, 560 and 1120 are accepted (as a ``Profile`` field and in astra.toml); ``None`` is the default and
    the key absent loads as ``None``."""
    _require_fields()
    assert Profile(name="lean").visual_tokens is None, (
        "the default is None: send nothing (engine default 280)"
    )
    for value in ALLOWED:
        assert _profile(visual_tokens=value).visual_tokens == value
        try:
            got = load(_toml(tmp_path, f"visual_tokens = {value}\n")).active().visual_tokens
        except ValueError as exc:
            raise AssertionError(
                f"T-0131 A1: visual_tokens = {value} loads; refused with {exc}"
            ) from None
        assert got == value, (value, got)
    assert load(_toml(tmp_path, "")).active().visual_tokens is None
    assert _profile(visual_tokens=None).visual_tokens is None


@pytest.mark.parametrize(
    ("toml_value", "py_value"),
    [("70", 70), ("140", 140), ("0", 0), ('"1120"', "1120"), ("1121", 1121)],
    ids=["70", "140", "0", "str-1120", "1121"],
)
def test_visual_tokens_junk_refused(tmp_path: Path, toml_value: str, py_value: Any) -> None:
    """A1 (adversarial): 70 and 140 (budgets the engine supports but the task does not), 0, the string ``"1120"``
    (lax coercion would accept it) and 1121 are refused at load, and the refusal names 280, 560 and 1120."""
    _require_fields()
    with pytest.raises(ValueError) as excinfo:
        load(_toml(tmp_path, f"visual_tokens = {toml_value}\n"))
    message = str(excinfo.value)
    assert _names_allowed(message), (
        f"T-0131 A1: the refusal names 280, 560 and 1120; it said:\n{message}"
    )
    with pytest.raises(ValueError) as excinfo:
        Profile(name="lean", visual_tokens=py_value)
    assert _names_allowed(str(excinfo.value)), str(excinfo.value)
    # the too-wide direction: the same key with an allowed value loads
    assert load(_toml(tmp_path, "visual_tokens = 1120\n")).active().visual_tokens == 1120


@pytest.mark.parametrize("tools", ["off", "dry_run"])
def test_reground_needs_live(tmp_path: Path, tools: str) -> None:
    """A1 (adversarial: a silent no-op is the failure): ``reground = true`` with ``tools = off`` or ``dry_run`` is
    refused at load, naming ``live``; ``reground = true`` with ``live`` loads, and ``reground = false`` loads with
    every mode (the too-wide direction)."""
    _require_fields()
    with pytest.raises(ValueError) as excinfo:
        load(_toml(tmp_path, f'tools = "{tools}"\nreground = true\n'))
    message = str(excinfo.value)
    assert "live" in message, f"T-0131 A1: the refusal names live; it said:\n{message}"
    with pytest.raises(ValueError):
        Profile(name="lean", tools=tools, reground=True)  # type: ignore[arg-type]
    try:
        live = load(_toml(tmp_path, 'tools = "live"\nreground = true\n')).active()
    except ValueError as exc:
        raise AssertionError(
            f"T-0131 A1: live with reground = true loads; refused with {exc}"
        ) from None
    assert live.reground is True and live.tools == "live"
    for mode in ("off", "dry_run", "live"):
        assert _profile(tools=mode, reground=False).reground is False
    # the key absent with tools off: the default, accepted. T-0132 R2 (A1, human ruling 2026-10-22): ``reground``
    # is tri-state and an absent key is ``None`` (unset), never ``False``
    assert load(_toml(tmp_path, f'tools = "{tools}"\n')).active().reground is None


#: the exact ``Profile.model_fields`` set at BASE a2d9ac6 (T-0131's list); T-0132 group B adds
#: ``reground_visual_tokens`` (``test_field_set_gains_reground_visual_tokens``)
T0131_FIELDS = {
    "name",
    "stt_backend",
    "tts_backend",
    "talker",
    "aec",
    "vram_budget_gib",
    "services",
    "diffusion",
    "tools",
    "tool_loop",
    "max_tool_steps",
    "browser",
    "browser_viewport",
    "visual_tokens",
    "reground",
}
#: T-0134c A1: the live-feedback default's two profile keys (re-scopes R1, R2)
T0134C_FIELDS = ("live_feedback", "action_echo")
#: T-0135b A1: the thought-call recovery profile key (T-0135b re-scopes R1, R2)
T0135B_FIELDS = ("thought_call_recovery",)
#: T-0140 re-pin (O-A2, docs/tasks/T-0140.md A5): ``Profile.page_overview``
T0140_FIELDS = ("page_overview",)
#: T-0150 re-pins R4/R5 (docs/tasks/T-0150.md Q-2, A6): ``Profile.auto_overview`` may be added (T-0150's own rows pin
#: it exactly); the rows below allow it and nothing else, green before and after T-0150
#: T-0151 re-pins R4/R5 (docs/tasks/T-0151.md Q-2, A2-9): ``Profile.overview_frame`` may be added the same way (T-0151's
#: own rows pin it exactly); green before and after T-0151 A2
T0150_FIELDS = ("auto_overview", "overview_frame")
#: T-0136 re-pins R1/R2 (docs/tasks/T-0136.md C-1, Q-2): ``Profile.action_outcome`` may be added the same way (T-0136's
#: own rows in ``test_t0136_cli.py`` pin it exactly); green before and after T-0136 group C
T0136_FIELDS = ("action_outcome",)
#: T-0137 re-pin (O-B2, docs/tasks/T-0137.md B-9): ``Profile.repeat_guard`` may be added the same way (T-0137's own rows
#: in ``test_t0137_cli.py`` pin it exactly); green before and after T-0137 group B
T0137_FIELDS = ("repeat_guard",)  # T-0137 re-pin
#: T-0125 pin move (docs/tasks/T-0125.md, Pins, amendment 2): ``Profile.search`` (default ``None``) is added the same
#: way; its values are pinned by ``test_config_search.py``
T0125_FIELDS = ("search",)  # T-0125 pin move
#: T-0160 re-pin (C-1, the T-0137 way): ``Profile.browser_web`` may be added (pinned exactly by
#: ``test_cli_tools_web.py``); green before and after T-0160 group C
T0160_FIELDS = ("browser_web",)  # T-0160 re-pin
T0172_FIELDS = ("announced_call_reask",)  # T-0172 R5 pin move (B9)
T0174_FIELDS = ("live_guidance",)  # T-0174 R-A5 pin move (A3)
T0178_FIELDS = ("max_tool_steps_web",)  # T-0178 A7 pin move (A1)
T0227_FIELDS = ("max_frames_per_turn", "frame_interval_ms")  # T-0227 R-2 re-pin (A-1)


def _require_t0227_fields() -> None:
    """T-0227 R-2 re-pin: exactly ``max_frames_per_turn`` and ``frame_interval_ms``, ``int | None = None`` each and
    ``None`` in every built-in profile."""
    from tests.unit.test_t0133_live_reground_off import FRAME_SETTING_FIELDS, _fields

    now = _fields(Profile, "local_astra.config")
    assert {k: now.get(k) for k in T0227_FIELDS} == FRAME_SETTING_FIELDS, (
        "T-0227 A-1: the two frame fields"
    )
    for name, profile in load(None).profiles.items():
        assert all(getattr(profile, k) is None for k in T0227_FIELDS), name


def _require_t0178_field() -> None:
    """T-0178 A7 (A1): ``max_tool_steps_web`` exists as ``int``, default 20, and is 20 in every built-in profile."""
    f = Profile.model_fields.get("max_tool_steps_web")
    assert f is not None, "T-0178 A1: Profile.max_tool_steps_web exists"
    assert f.annotation is int and f.default == 20 and not f.is_required(), f
    for name, profile in load(None).profiles.items():
        assert profile.max_tool_steps_web == 20, name  # type: ignore[attr-defined]


T0182_FIELDS = ("filler_clips", "still_working_after_s")  # T-0182 B-R3 pin move (B2, B4)


def _require_t0182_fields() -> None:
    """T-0182 B2/B4: ``filler_clips: bool = False`` and ``still_working_after_s: int = 20`` exist, at those values in
    every built-in profile (their validation is pinned by ``test_t0182_cli``)."""
    fc = Profile.model_fields.get("filler_clips")
    assert fc is not None, "T-0182 B2: Profile.filler_clips exists"
    assert fc.annotation is bool and fc.default is False and not fc.is_required(), fc
    sw = Profile.model_fields.get("still_working_after_s")
    assert sw is not None, "T-0182 B4: Profile.still_working_after_s exists"
    assert sw.annotation is int and sw.default == 20 and not sw.is_required(), sw
    for name, profile in load(None).profiles.items():
        assert profile.filler_clips is False, name  # type: ignore[attr-defined]
        assert profile.still_working_after_s == 20, name  # type: ignore[attr-defined]


def _require_t0174_field() -> None:
    """T-0174 R-A5 (A3): ``live_guidance`` exists as ``str | None = None``, ``None`` in every built-in profile."""
    f = Profile.model_fields.get("live_guidance")
    assert f is not None, "T-0174 A3: Profile.live_guidance exists"
    assert f.annotation == (str | None) and f.default is None and not f.is_required(), f
    for name, profile in load(None).profiles.items():
        assert profile.live_guidance is None, name  # type: ignore[attr-defined]


def test_defaults_unchanged(tmp_path: Path) -> None:
    """A1/A5: the new fields default to ``None`` in every built-in profile, and every other default is BASE's
    (written out from ``config.py`` at f30365d).

    T-0132 R2 (A1, human ruling 2026-10-22): ``reground`` defaults to ``None`` (unset; the live default is resolved
    in ``cli``), not ``False``. The field set is T-0131's apart from group B's ``reground_visual_tokens``, whose
    presence is pinned exactly by ``test_field_set_gains_reground_visual_tokens`` (so group A can be activated on its
    own)."""
    _require_fields()
    cfg = load(None)
    for name, profile in cfg.profiles.items():
        assert profile.visual_tokens is None, name
        assert profile.reground is None, f"T-0132 R2: {name}'s reground defaults to None (unset)"
        assert (
            profile.tools == "off" and profile.tool_loop is None and profile.max_tool_steps == 8
        ), name
        assert profile.browser == "headed" and profile.browser_viewport == (1280, 800), name
    lean = cfg.active()
    assert lean.name == "lean" and lean.diffusion.canvas_length == 256
    assert lean.diffusion.max_denoising_steps == 48 and lean.diffusion.thinking is True
    # T-0134c re-scope R1 (docs/tasks/T-0134c.md A1): T-0134c's two fields are subtracted too (pinned exactly by
    # ``test_field_set_gains_reground_visual_tokens``), so this row stays about T-0131's additions. T-0135b re-scope
    # R1 (docs/tasks/T-0135b.md A1): so is ``thought_call_recovery`` (green before and after T-0135b)
    assert (
        set(Profile.model_fields)
        - {
            "reground_visual_tokens",
            *T0134C_FIELDS,
            *T0135B_FIELDS,
            *T0140_FIELDS,
            *T0150_FIELDS,
            *T0136_FIELDS,  # T-0136 re-pin R1
            *T0137_FIELDS,  # T-0137 re-pin
            *T0125_FIELDS,  # T-0125 pin move
            *T0160_FIELDS,  # T-0160 re-pin
            *T0172_FIELDS,  # T-0172 R5 pin move (required present, below)
            *T0174_FIELDS,  # T-0174 R-A5 pin move (required present, below)
            *T0178_FIELDS,  # T-0178 A7 pin move (required present, below)
            *T0182_FIELDS,  # T-0182 B-R3 pin move (required present, below)
            *T0227_FIELDS,  # T-0227 R-2 re-pin (required present, below)
        }
        == T0131_FIELDS
    ), "A1 adds exactly visual_tokens and reground"
    _require_t0174_field()  # T-0174 R-A5 pin move
    _require_t0178_field()  # T-0178 A7 pin move
    _require_t0182_fields()  # T-0182 B-R3 pin move
    _require_t0227_fields()  # T-0227 R-2 re-pin
    for name, profile in load(None).profiles.items():  # T-0125 pin move: present, default None
        assert profile.search is None, name  # type: ignore[attr-defined]
    assert load(_toml(tmp_path, "")).active() == Profile(name="lean")


def test_field_set_gains_reground_visual_tokens() -> None:
    """T-0132 R2, group B (B1): the exact field set is T-0131's plus ``reground_visual_tokens``, defaulting to
    ``None`` in every built-in profile.

    T-0134c re-scope R2 (docs/tasks/T-0134c.md A1, "Profile gains two fields"): the exact set also holds
    ``live_feedback`` and ``action_echo``, each ``None`` in every built-in profile. Red until T-0134c is
    implemented, by design.

    T-0135b re-scope R2 (docs/tasks/T-0135b.md A1, "Profile gains thought_call_recovery"): the exact set also holds
    ``thought_call_recovery``, ``None`` in every built-in profile. Red until T-0135b is implemented, by design."""
    # T-0140 re-pin (O-A2): the exact set also holds ``page_overview``, ``None`` in every built-in profile
    want = T0131_FIELDS | {"reground_visual_tokens", *T0134C_FIELDS, *T0135B_FIELDS, *T0140_FIELDS}
    got = set(Profile.model_fields) - set(
        T0150_FIELDS
    )  # T-0150 re-pin R5: only auto_overview may be added
    got -= set(T0136_FIELDS)  # T-0136 re-pin R2: and action_outcome
    got -= set(T0137_FIELDS)  # T-0137 re-pin: and repeat_guard
    assert set(T0125_FIELDS) <= got, "T-0125 pin move: Profile.search exists"
    got -= set(T0125_FIELDS)  # T-0125 pin move: and search
    got -= set(T0160_FIELDS)  # T-0160 re-pin: and browser_web when present
    # T-0172 R5 pin move (B9): announced_call_reask must exist (bool | None = None, None in every built-in profile)
    assert set(T0172_FIELDS) <= got, "T-0172 B9: Profile.announced_call_reask exists"
    for name, profile in load(None).profiles.items():
        assert profile.announced_call_reask is None, name  # type: ignore[attr-defined]
    got -= set(T0172_FIELDS)
    # T-0174 R-A5 pin move (A3): live_guidance must exist (str | None = None, None in every built-in profile)
    _require_t0174_field()
    got -= set(T0174_FIELDS)
    _require_t0178_field()  # T-0178 A7 pin move: exactly max_tool_steps_web, int = 20
    got -= set(T0178_FIELDS)
    _require_t0182_fields()  # T-0182 B-R3 pin move: exactly filler_clips and still_working_after_s
    got -= set(T0182_FIELDS)
    _require_t0227_fields()  # T-0227 R-2 re-pin: exactly max_frames_per_turn and frame_interval_ms
    got -= set(T0227_FIELDS)
    assert got == want, sorted(got ^ want)
    present = tuple(
        f for f in (*T0150_FIELDS, *T0136_FIELDS, *T0137_FIELDS) if f in Profile.model_fields
    )  # T-0136 re-pin R2; T-0137 re-pin
    for name, profile in load(None).profiles.items():
        assert profile.reground_visual_tokens is None, name  # type: ignore[attr-defined]
        for field in (*T0134C_FIELDS, *T0135B_FIELDS, *T0140_FIELDS, *present):
            assert getattr(profile, field) is None, (name, field)
