"""T-0114 criterion 14, the config half: ``Profile.tool_loop`` and ``Profile.max_tool_steps``.

Spec (docs/tasks/T-0114.md criterion 14): ``Profile.tool_loop: bool = False`` and ``Profile.max_tool_steps: int`` with
default 8 and ``ge=1``, so ``0`` and ``-1`` raise at load. The shipped profiles (``config.DEFAULTS`` and the repo's
``astra.toml``) still load, with the loop off.

Re-scoped by T-0123 criterion 1 (H-T123-1): ``Profile.tool_loop: bool | None = None``, ``None`` (the key absent)
meaning "on when tools are on"; ``true``/``false`` load as bools; the shipped profiles load with ``None``. The
re-scoped rows are active since T-0123's activation.

``Profile`` is a ``Contract`` (``extra="forbid"``), so *any* unknown key already raises a ``ValidationError`` before
T-0114. The ``ge=1`` rows therefore check the error's ``type`` and ``loc``, not merely that something raised; otherwise
they would pass with the field absent.

Regime: CPU only, no I/O beyond ``tmp_path``; no measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from local_astra.config import Profile, load

REPO = Path(__file__).resolve().parents[2]

#: criterion 14, written out from the task text; T-0123 criterion 1 (H-T123-1): ``None`` is "on when tools are on"
DEFAULT_TOOL_LOOP = None
DEFAULT_MAX_TOOL_STEPS = 8


def _write(tmp_path: Path, **lean: Any) -> Path:
    lines = ["profile = 'lean'", "[profiles.lean]"]
    for key, value in lean.items():
        rendered = ("true" if value else "false") if isinstance(value, bool) else repr(value)
        lines.append(f"{key} = {rendered}")
    path = tmp_path / "astra.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _need_fields() -> None:
    """A missing field is a named assertion, not an ``extra_forbidden`` error or an ``AttributeError``."""
    missing = {"tool_loop", "max_tool_steps"} - set(Profile.model_fields)
    assert not missing, f"T-0114 criterion 14: Profile lacks {sorted(missing)}"


def _ge_errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        e
        for e in exc.errors()
        if e.get("loc")
        and e["loc"][-1] == "max_tool_steps"
        and e.get("type") == "greater_than_equal"
    ]


def test_profile_declares_tool_loop_as_optional_bool_defaulting_to_none():
    """Criterion 14 as re-scoped by T-0123 criterion 1 (H-T123-1): ``Profile.tool_loop: bool | None = None``."""
    fields = Profile.model_fields
    assert "tool_loop" in fields, (
        f"T-0114 criterion 14: Profile.tool_loop exists; fields are {sorted(fields)}"
    )
    assert fields["tool_loop"].annotation == (bool | None), fields["tool_loop"].annotation
    assert Profile(name="lean").tool_loop is DEFAULT_TOOL_LOOP


def test_profile_declares_max_tool_steps_defaulting_to_8():
    """Criterion 14: ``Profile.max_tool_steps: int``, default 8."""
    fields = Profile.model_fields
    assert "max_tool_steps" in fields, (
        f"T-0114 criterion 14: Profile.max_tool_steps exists; fields are {sorted(fields)}"
    )
    assert fields["max_tool_steps"].annotation is int, fields["max_tool_steps"].annotation
    assert Profile(name="lean").max_tool_steps == DEFAULT_MAX_TOOL_STEPS


@pytest.mark.parametrize("bad", [0, -1])
def test_max_tool_steps_below_1_raises_at_load(tmp_path, bad):
    """Criterion 14: ``ge=1`` — ``0`` and ``-1`` raise at load, as a ``greater_than_equal`` error on the field
    (an unknown-field error, which is what an absent field gives, does not count)."""
    with pytest.raises(ValidationError) as caught:
        load(_write(tmp_path, max_tool_steps=bad))
    assert _ge_errors(caught.value), (
        f"T-0114 criterion 14: max_tool_steps = {bad} is refused by ge=1; the errors were {caught.value.errors()}"
    )
    with pytest.raises(ValidationError) as direct:
        Profile(name="lean", max_tool_steps=bad)
    assert _ge_errors(direct.value), direct.value.errors()


def test_the_bound_is_not_too_wide_1_and_a_set_loop_load(tmp_path):
    """Criterion 14, the guard's other side: ``max_tool_steps = 1`` is the smallest legal value and loads, and the
    two keys are settable from ``astra.toml`` (a bound of ``gt=1`` or ``ge=2`` fails here). T-0123 criterion 1: an
    absent ``tool_loop`` loads as ``None`` and ``false`` as ``False`` (never coerced to one another)."""
    _need_fields()
    profile = load(_write(tmp_path, tool_loop=True, max_tool_steps=1)).active()
    assert profile.tool_loop is True
    assert profile.max_tool_steps == 1
    other = load(_write(tmp_path, max_tool_steps=3)).active()
    assert other.max_tool_steps == 3 and other.tool_loop is None
    opted_out = load(_write(tmp_path, tool_loop=False)).active()
    assert opted_out.tool_loop is False, opted_out.tool_loop


@pytest.mark.parametrize("source", ["defaults", "repo astra.toml"])
def test_the_shipped_profiles_load_with_the_loop_unset_and_8_steps(source):
    """Criterion 14 / T-0123 criterion 1: every shipped profile still loads, with ``tool_loop`` unset (``None``)
    and the default bound (none of them sets either key)."""
    _need_fields()
    cfg = load(None) if source == "defaults" else load(REPO / "astra.toml")
    assert cfg.profiles, "premise: the shipped config has profiles"
    for name, profile in cfg.profiles.items():
        assert profile.tool_loop is DEFAULT_TOOL_LOOP, (name, profile.tool_loop)
        assert profile.max_tool_steps == DEFAULT_MAX_TOOL_STEPS, (name, profile.max_tool_steps)


def test_the_shipped_astra_toml_sets_neither_loop_key():
    """Criterion 14 ("the shipped profiles load unchanged"): the repo's ``astra.toml`` is not edited to turn the
    loop on or move its bound; with the loop off by default, no shipped profile opts in. Holds now."""
    import tomllib

    with (REPO / "astra.toml").open("rb") as handle:
        doc = tomllib.load(handle)
    for name, body in doc.get("profiles", {}).items():
        assert "tool_loop" not in body and "max_tool_steps" not in body, (name, sorted(body))
