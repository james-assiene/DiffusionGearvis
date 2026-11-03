"""astra.toml profiles and the loopback rule (docs/02 §7, §9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from local_astra.cli import main
from local_astra.config import Profile, Services, load

REPO = Path(__file__).resolve().parents[2]


def test_defaults_load_without_a_file():
    cfg = load(None)
    assert cfg.profile == "lean"
    assert cfg.active().diffusion.canvas_length == 256
    assert cfg.active().diffusion.max_num_seqs == 3
    assert cfg.active().diffusion.thinking is True
    assert set(cfg.profiles) >= {"lean", "cpu-stt", "quality-tts", "talker"}


def test_repo_astra_toml_loads_and_matches_defaults():
    cfg = load(REPO / "astra.toml")
    assert cfg.profile == "lean"
    assert cfg.profiles["talker"].talker is True
    assert cfg.profiles["talker"].services.talker == "http://127.0.0.1:8003"
    assert cfg.profiles["cpu-stt"].stt_backend == "nemotron-cpu"


def test_user_file_overrides_merge_with_defaults(tmp_path):
    p = tmp_path / "astra.toml"
    p.write_text('profile = "cpu-stt"\n[profiles.cpu-stt.diffusion]\nmax_denoising_steps = 24\n')
    cfg = load(p)
    assert cfg.profile == "cpu-stt"
    assert cfg.active().diffusion.max_denoising_steps == 24
    assert cfg.active().stt_backend == "nemotron-cpu"  # default for that profile survives
    assert cfg.profiles["lean"].diffusion.max_denoising_steps == 48


@pytest.mark.parametrize("field", ["vllm", "stt", "tts"])
def test_non_loopback_endpoints_are_rejected(field):
    with pytest.raises(ValidationError):
        Services(**{field: "http://192.168.1.10:8001"})


def test_max_num_seqs_is_capped_at_four():
    with pytest.raises(ValidationError):
        Profile(name="lean", diffusion={"max_num_seqs": 5})  # type: ignore[arg-type]


def test_cli_config_prints_the_active_profile(capsys):
    assert main(["--config", str(REPO / "astra.toml"), "config"]) == 0
    out = capsys.readouterr().out
    assert '"name": "lean"' in out and '"canvas_length": 256' in out


def test_cli_unimplemented_commands_exit_nonzero(capsys):
    # `status` became real under T-0033 (ruling A13); `fetch-models` is still the scaffold stub.
    assert main(["--config", str(REPO / "astra.toml"), "fetch-models"]) == 2
    assert "not implemented" in capsys.readouterr().err


# --- T-0092b criterion 1: the `tools` switch is a profile setting, `off` by default (H3) ---------

#: What a refusal must name (criterion 1): the v1 non-goal and the umbrella that decided it.
REFUSAL_NAMES = ("docs/01 §3", "T-0080d")


def _tools_field() -> object:
    field = Profile.model_fields.get("tools")
    assert field is not None, "T-0092b criterion 1: Profile.tools exists"
    return field


def test_tools_is_off_by_default_in_every_profile():
    """Criterion 1 / H3: ``Profile.tools`` defaults to ``off``; no shipped profile turns it on."""
    field = _tools_field()
    assert not field.is_required() and field.default == "off", (  # type: ignore[attr-defined]
        f"T-0092b criterion 1: Profile.tools defaults to 'off'; it is {field!r}"
    )
    for cfg in (load(None), load(REPO / "astra.toml")):
        assert {name: p.tools for name, p in cfg.profiles.items()} == {
            name: "off" for name in cfg.profiles
        }, "H3: every profile ships with tools = off until T-0093 passes"


def test_dry_run_is_settable_in_astra_toml(tmp_path):
    """Criterion 1: ``tools = "dry_run"`` in a profile table loads, and the other profiles stay off."""
    _tools_field()
    p = tmp_path / "astra.toml"
    p.write_text('profile = "lean"\n[profiles.lean]\ntools = "dry_run"\n')
    cfg = load(p)
    assert cfg.active().tools == "dry_run"
    assert cfg.profiles["cpu-stt"].tools == "off", (
        "setting one profile leaves the others at the default"
    )


#: T-0111g re-scope R1: ``"live"`` left this list when ``tools = "live"`` became a mode (docs/tasks/T-0111g.md B1,
#: "Named re-scopes"); its acceptance is ``tests/unit/test_config_tools_live.py``. The other values keep their rows.
@pytest.mark.parametrize("value", ["junk", "dry-run", "on", "", "Off"])
def test_any_other_value_is_refused_at_config_load_naming_the_non_goal(tmp_path, value):
    """Criterion 1: any value but the modes is refused at load, naming docs/01 §3 and T-0080d.

    ``dry-run`` is the *argv* spelling, not the config one: the profile value is the Literal's ``dry_run``.
    """
    _tools_field()
    p = tmp_path / "astra.toml"
    p.write_text(f'profile = "lean"\n[profiles.lean]\ntools = "{value}"\n')
    with pytest.raises(ValueError) as excinfo:
        load(p)
    message = str(excinfo.value)
    for name in REFUSAL_NAMES:
        assert name in message, (
            f"T-0092b criterion 1: refusing tools={value!r} names {name!r}; the message was:\n{message}"
        )
    with pytest.raises(ValidationError):
        Profile(name="lean", tools=value)  # type: ignore[arg-type]


def test_the_two_accepted_values_are_exactly_off_and_dry_run():
    """Criterion 1, the too-wide direction: the guard must not refuse what the criterion allows."""
    _tools_field()
    assert Profile(name="lean", tools="off").tools == "off"  # type: ignore[call-arg]
    assert Profile(name="lean", tools="dry_run").tools == "dry_run"  # type: ignore[call-arg]
