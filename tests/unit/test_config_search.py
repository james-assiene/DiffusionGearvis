"""T-0125 criterion 2: ``Profile.search`` and its load-time refusals (docs/tasks/T-0125.md).

``Profile.search: Literal["off", "ddgs"] | None = None`` (``None``: the mode's default); any other value is refused
at load, and ``search = "ddgs"`` with ``tools = "off"`` is refused at load. Expected values are written out from the
criterion. Written ahead of T-0125 as pending rows (activated with the implementation): at BASE the field does not exist (an unknown key is either
refused for the wrong reason or ignored), so each fails on a named assertion.

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_astra.config import Profile, load


def _toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "astra.toml"
    p.write_text('profile = "lean"\n[profiles.lean]\n' + body)
    return p


def _field() -> typing.Any:
    assert "search" in Profile.model_fields, (
        "T-0125 criterion 2: Profile gains `search: Literal['off', 'ddgs'] | None = None`; fields "
        f"{sorted(Profile.model_fields)}"
    )
    return Profile.model_fields["search"]


def test_search_field_default_none_and_literal() -> None:
    field = _field()
    assert field.default is None, f"T-0125 criterion 2: the default is None; {field.default!r}"
    assert Profile(name="lean").search is None  # type: ignore[attr-defined]
    args = set()
    for part in typing.get_args(field.annotation):
        args |= set(typing.get_args(part)) or {part}
    assert args == {"off", "ddgs", type(None)}, (
        f"T-0125 criterion 2: Literal['off', 'ddgs'] | None; {field.annotation!r}"
    )


@pytest.mark.parametrize(
    ("tools", "value"),
    [("dry_run", "ddgs"), ("live", "ddgs"), ("live", "off"), ("dry_run", "off"), ("off", "off")],
)
def test_search_values_accepted(tmp_path: Path, tools: str, value: str) -> None:
    _field()
    try:
        cfg = load(_toml(tmp_path, f'tools = "{tools}"\nsearch = "{value}"\n'))
    except ValueError as exc:
        raise AssertionError(
            f"T-0125 criterion 2: search={value!r} with tools={tools!r} loads; {exc}"
        ) from None
    assert cfg.active().search == value  # type: ignore[attr-defined]
    assert load(_toml(tmp_path, f'tools = "{tools}"\n')).active().search is None  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "value", ['"on"', '"duckduckgo"', "true", '"DDGS"', '""', '"brave-api"', "1"]
)
def test_unknown_search_values_refused(tmp_path: Path, value: str) -> None:
    _field()
    with pytest.raises((ValueError, ValidationError)) as excinfo:
        load(_toml(tmp_path, f'tools = "live"\nsearch = {value}\n'))
    message = str(excinfo.value)
    assert "search" in message, f"T-0125 criterion 2: the refusal names the key; {message}"
    with pytest.raises(ValidationError):
        Profile(name="lean", search=value.strip('"'))  # type: ignore[call-arg]


def test_ddgs_with_tools_off_refused(tmp_path: Path) -> None:
    """``search = "ddgs"`` with ``tools = "off"`` set explicitly in the same profile is refused at load, naming both
    (amendment 2: the ``reground_visual_tokens`` pattern). A profile with no ``tools`` key loads: the CLI resolves
    it (``--tools`` may turn tools on), and ``test_cli_search`` pins its refusal there."""
    _field()
    with pytest.raises(ValueError) as excinfo:
        load(_toml(tmp_path, 'tools = "off"\nsearch = "ddgs"\n'))
    message = str(excinfo.value)
    assert "search" in message and "tools" in message, (
        f"T-0125 criterion 2: the refusal names search and tools; {message}"
    )
    try:
        unset = load(_toml(tmp_path, 'search = "ddgs"\n')).active()
    except ValueError as exc:
        raise AssertionError(f"T-0125 amendment 2: no tools key loads; {exc}") from None
    assert unset.search == "ddgs"  # type: ignore[attr-defined]
    # the too-wide direction: off with tools off loads
    assert load(_toml(tmp_path, 'tools = "off"\nsearch = "off"\n')).active().search == "off"  # type: ignore[attr-defined]
