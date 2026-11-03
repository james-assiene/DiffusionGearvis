"""T-0111g B1: ``tools = "live"`` in the config, the browser fields, and live-without-loop refused at load.

Expected values are written out from the criterion. Every row but the junk row is ``pending(task="T-0111g")``: at
BASE ``live`` is refused and the fields do not exist, so each fails on a named assertion. The junk row is green at
BASE (its message already names the three values) and must stay green, so it carries no marker.

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest
from pydantic import ValidationError

from local_astra import config
from local_astra.config import Profile, load


def _toml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "astra.toml"
    p.write_text('profile = "lean"\n[profiles.lean]\n' + body)
    return p


def test_live_accepted(tmp_path: Path) -> None:
    """B1: ``live`` is a tools mode, in the Literal and in ``TOOLS_MODES``, settable in astra.toml."""
    assert set(typing.get_args(config.ToolsMode)) == {"off", "dry_run", "live"}, config.ToolsMode
    assert tuple(config.TOOLS_MODES) == ("off", "dry_run", "live") or set(config.TOOLS_MODES) == {
        "off",
        "dry_run",
        "live",
    }, config.TOOLS_MODES
    try:
        profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    except ValidationError as exc:
        raise AssertionError(f"T-0111g B1: tools='live' is accepted; refused with {exc}") from None
    assert profile.tools == "live"
    cfg = load(_toml(tmp_path, 'tools = "live"\n'))
    assert cfg.active().tools == "live"
    assert cfg.profiles["cpu-stt"].tools == "off", "the other profiles stay off"


def test_live_with_loop_false_refused(tmp_path: Path) -> None:
    """B1 (mutation 6): ``tools = "live"`` with ``tool_loop = false`` is refused at load, naming the loop; with the
    loop unset or true it loads (the too-wide direction)."""
    with pytest.raises(ValueError) as excinfo:
        load(_toml(tmp_path, 'tools = "live"\ntool_loop = false\n'))
    message = str(excinfo.value)
    assert "tool_loop" in message or "loop" in message.lower(), message
    assert "live" in message, message
    try:
        assert load(_toml(tmp_path, 'tools = "live"\n')).active().tool_loop in (None, True)
        assert (
            load(_toml(tmp_path, 'tools = "live"\ntool_loop = true\n')).active().tool_loop is True
        )
    except ValueError as exc:
        raise AssertionError(f"T-0111g B1: live with the loop on/unset loads; {exc}") from None
    assert (
        load(_toml(tmp_path, 'tools = "dry_run"\ntool_loop = false\n')).active().tool_loop is False
    ), "dry_run with the loop off stays allowed (T-0114)"


@pytest.mark.parametrize("value", ["junk", "dry-run", "on", "", "Off", "LIVE"])
def test_junk_still_names_docs01_and_t0080d(tmp_path: Path, value: str) -> None:
    """B1: ``tools_refusal`` names the three modes and still names docs/01 §3 and T-0080d."""
    with pytest.raises(ValueError) as excinfo:
        load(_toml(tmp_path, f'tools = "{value}"\n'))
    message = str(excinfo.value)
    for name in ("docs/01 §3", "T-0080d", "'off'", "'dry_run'", "'live'"):
        assert name in message, (
            f"T-0111g B1: refusing tools={value!r} names {name!r}; said:\n{message}"
        )
    text = config.tools_refusal(value)
    for name in ("'off'", "'dry_run'", "'live'"):
        assert name in text, text


def test_browser_fields_defaults(tmp_path: Path) -> None:
    """B1 (H-T111c-1, H-T111c-3): ``browser`` defaults to headed, takes headed|headless only; ``browser_viewport``
    defaults to (1280, 800)."""
    fields = Profile.model_fields
    assert "browser" in fields and "browser_viewport" in fields, sorted(fields)
    p = Profile(name="lean")
    assert p.browser == "headed"  # type: ignore[attr-defined]
    assert tuple(p.browser_viewport) == (1280, 800)  # type: ignore[attr-defined]
    cfg = load(_toml(tmp_path, 'browser = "headless"\nbrowser_viewport = [1024, 768]\n'))
    assert cfg.active().browser == "headless"  # type: ignore[attr-defined]
    assert tuple(cfg.active().browser_viewport) == (1024, 768)  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        Profile(name="lean", browser="kiosk")  # type: ignore[call-arg]
