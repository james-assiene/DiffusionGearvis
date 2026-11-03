"""T-0151 A2-9 (cli, config): the opt-in, its source, the refusals, the wiring and the startup line
(docs/tasks/T-0151.md).

**The surface these rows pin** (the task's names; readings stated):

- ``config.LIVE_DEFAULT_OVERVIEW_FRAME = False``; ``Profile.overview_frame: bool | None = None``; TOML booleans only
  (a non-bool value is a load refusal); refused at load: the key with ``tools != "live"`` (naming ``live``, as T-0150's
  key), and ``overview_frame = true`` together with ``auto_overview = true``, ``page_overview = true`` or
  ``reground = true`` (the message names ``overview_frame``).
- ``astra session --overview-frame`` / ``--no-overview-frame``; argparse dest ``args.overview_frame`` (read with
  ``getattr``). ``cli.overview_frame_of(profile, args, mode) -> OverviewFrame(overview_frame, overview_frame_from)``,
  read as a two-field value in that order (``pair``); sources ``flag``/``profile``/``default``; outside live only
  "not on" is pinned.
- Exit 2 before anything is built: either flag outside a resolved ``live`` mode (naming ``live``); the key of a live
  profile whose mode ``--tools`` overrode (naming ``live``); the overview frame on together with the auto overview,
  the page overview or ReGround from any source pair, naming both (read: once ``_`` and ``-`` are spaces, the text
  holds "overview frame" and "auto overview" / "page overview" / "reground").
- ``Session`` gets ``overview_frame=True`` only when on; off passes no keyword. ``tools_version`` stays v1.
- The line, live only: ``astra session: overview frame: off (default)`` / ``on (flag)`` / ``on (profile)``, right after
  the ``auto overview`` line and before the regime line (``--dry-run`` too, where there is no regime line).

Readings: **C1** (ruled O-A2, "accepted"): only ``overview_frame = true`` beside ``auto_overview``/``page_overview``/
``reground = true`` is a load refusal; ``overview_frame = false`` beside each of them loads
(``test_false_key_beside_others_loads``). **C2** the
load refusals' wording is not pinned beyond naming ``overview_frame`` (and ``live`` for the mode refusal).

Regime: CPU, the cli rigs over fakes, nothing served, n = 1. BASE_A2 = 177c0a1. Activated for group A2
(rulings O-A2 in the task file): every row is red until group A2's cli and config are implemented.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

import pytest

from local_astra import cli, config
from local_astra.config import Profile, load
from tests.unit import _t0202_pins as t0202
from tests.unit.test_cli_t0131 import MODE_FLAGS, dry_run
from tests.unit.test_cli_t0132 import _toml, build
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0134c_live_feedback_default import _names_live
from tests.unit.test_t0140_cli import pair, session_help
from tests.unit.test_t0150_cli import refused_run

V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
PREFIX = "astra session: overview frame: "
AUTO_PREFIX = "astra session: auto overview: "
ON, OFF = "--overview-frame", "--no-overview-frame"
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'
LIVE = MODE_FLAGS["live"]


def require() -> Any:
    flags = set(re.findall(r"--[a-z][a-z-]+", session_help()))
    missing = [f for f in (ON, OFF) if f not in flags]
    assert not missing, f"T-0151 A2-9: astra session {missing} do not exist yet"
    assert "overview_frame" in Profile.model_fields, (
        "T-0151 A2-9: Profile.overview_frame does not exist yet"
    )
    fn = getattr(cli, "overview_frame_of", None)
    assert callable(fn), (
        "T-0151 A2-9: cli.overview_frame_of(profile, args, mode) does not exist yet"
    )
    return fn


def lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith(PREFIX)]


def _spaced(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def names_both(text: str, other: str) -> bool:
    low = _spaced(text)
    return "overview frame" in low and other in low


# =====================================================================================================
# config
# =====================================================================================================


def test_default_on(tmp_path, capsys, monkeypatch):
    """A2-9, as re-pinned by T-0152 (R1; was ``test_default_off``): the constant is ``True``; the key defaults to
    ``None`` (annotation ``bool | None``) in every built-in profile; the resolver says ``(True, "default")``; the
    default build passes ``overview_frame=True``; the default live line reads ``on (default)``."""
    fn = require()
    # T-0152 re-pin R1: the default is on (docs/tasks/T-0152.md A1/A2/A4/A5)
    assert getattr(config, "LIVE_DEFAULT_OVERVIEW_FRAME", None) is True
    f = Profile.model_fields["overview_frame"]
    assert f.default is None and f.annotation == (bool | None), f
    for name, p in load(None).profiles.items():
        assert p.overview_frame is None, name  # type: ignore[attr-defined]
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    assert pair(fn(profile, argparse.Namespace(), "live")) == (True, "default")  # T-0152 re-pin R1
    calls, _ = build(tmp_path / "b", capsys, monkeypatch)
    assert calls.session_kwargs[-1].get("overview_frame") is True, (  # T-0152 re-pin R1
        "T-0152 A4: the default build passes Session(overview_frame=True)"
    )
    monkeypatch.undo()
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live")
    assert lines(out) == [PREFIX + "on (default)"], out  # T-0152 re-pin R1


def _load(tmp_path: Any, name: str, body: str) -> Any:
    d = tmp_path / name
    d.mkdir()
    return load(_toml(d, body))


def test_profile_key_load_refusals(tmp_path):
    """A2-9: the key with ``tools`` other than ``live`` (off, dry_run, absent) is refused at load naming ``live``; a
    non-bool value is refused; ``overview_frame = true`` with ``auto_overview = true``, ``page_overview = true`` or
    ``reground = true`` is refused (reading C1/C2). Too-wide companions: the key alone (true, false) and true beside each
    of the three set false load."""
    require()
    for i, tools_line in enumerate(('tools = "off"\n', 'tools = "dry_run"\n', "")):
        for v in ("true", "false"):
            with pytest.raises(ValueError) as exc:
                _load(tmp_path, f"t{i}-{v}", f"{tools_line}overview_frame = {v}\n")
            assert _names_live(str(exc.value)) and "overview_frame" in str(exc.value), (
                tools_line,
                v,
                str(exc.value),
            )
    for j, junk in enumerate(('"on"', '"true"', "1", "0", '"yes"')):
        with pytest.raises(ValueError) as exc:
            _load(tmp_path, f"junk-{j}", f'tools = "live"\noverview_frame = {junk}\n')
        assert "overview_frame" in str(exc.value), (junk, str(exc.value))
    for other in ("auto_overview", "page_overview", "reground"):
        with pytest.raises(ValueError) as exc:
            _load(
                tmp_path,
                f"both-{other}",
                f'tools = "live"\noverview_frame = true\n{other} = true\n',
            )
        assert "overview_frame" in str(exc.value), (other, str(exc.value))
    for k, (body, want) in enumerate(
        (
            ('tools = "live"\noverview_frame = true\n', True),
            ('tools = "live"\noverview_frame = false\n', False),
            ('tools = "live"\noverview_frame = true\nauto_overview = false\n', True),
            ('tools = "live"\noverview_frame = true\npage_overview = false\n', True),
            ('tools = "live"\noverview_frame = true\nreground = false\n', True),
        )
    ):
        assert _load(tmp_path, f"ok-{k}", body).active().overview_frame is want, body  # type: ignore[attr-defined]


# =====================================================================================================
# the resolver, the wiring, the line
# =====================================================================================================

SOURCES = {
    "flag-on": ([ON], "", (True, "flag")),
    "flag-off": ([OFF], "", (False, "flag")),
    "profile-true": ([], "overview_frame = true\n", (True, "profile")),
    "profile-false": ([], "overview_frame = false\n", (False, "profile")),
    "flag-off-over-profile": ([OFF], "overview_frame = true\n", (False, "flag")),
    "flag-on-over-profile": ([ON], "overview_frame = false\n", (True, "flag")),
}


@pytest.mark.parametrize("case", list(SOURCES))
def test_sources_and_wiring(tmp_path, capsys, monkeypatch, case):
    """A2-9: flag, else key, else default, each with its source; the dry-run line says it; the built ``Session`` gets
    ``overview_frame=True`` only when on (absent when off); the kit's ``tools_version`` stays v1's either way."""
    fn = require()
    flags, body, (value, src) = SOURCES[case]
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body)
    assert lines(out) == [PREFIX + ("on" if value else "off") + f" ({src})"], out
    monkeypatch.undo()
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    kw = calls.session_kwargs[-1]
    if value:
        assert kw.get("overview_frame") is True, (
            f"T-0151 A2-9: Session(overview_frame=True); {sorted(kw)}"
        )
    else:
        assert "overview_frame" not in kw, f"T-0151 A2-9: off passes no keyword; {sorted(kw)}"
    # T-0202 D3 re-pin: the tools stay the default live set (v1, the wait set once the wait tool is on by default)
    assert calls.kit().tools_version == t0202.default_live_sha(), "T-0151 A2-9: the tools stay v1"
    assert t0202.default_on() or calls.kit().tools_version == V1_SHA
    (tmp_path / "p").mkdir()
    profile = load(_toml(tmp_path / "p", LIVE_PROFILE + body)).active()
    flag = True if ON in flags else False if OFF in flags else None
    assert pair(fn(profile, argparse.Namespace(overview_frame=flag), "live")) == (value, src)


def test_resolver_not_on_outside_live() -> None:
    """A2-9: outside a resolved mode of live the resolver is never on (only "not on" is pinned)."""
    fn = require()
    for mode in ("off", "dry_run"):
        value, _ = pair(fn(Profile(name="lean"), argparse.Namespace(overview_frame=None), mode))
        assert not value, mode


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("mode", ["off", "dry_run"])
def test_resolver_ignores_flag_outside_live(mode, flag) -> None:
    """A2-9 (review B; the mutant reads the flag before the ``mode != "live"`` return): the resolver itself, given
    ``args.overview_frame`` set in a resolved mode other than live, gives off from ``"unset"`` (the refusal of such a
    flag is ``session_cmd``'s, not the resolver's). Also with a live profile carrying the key."""
    fn = require()
    for profile in (Profile(name="lean"), Profile(name="lean", tools="live", overview_frame=True)):  # type: ignore[arg-type, call-arg]
        got = pair(fn(profile, argparse.Namespace(overview_frame=flag), mode))
        assert got == (False, "unset"), (mode, flag, profile.tools, got)


@pytest.mark.parametrize("other", ["auto_overview", "page_overview", "reground"])
def test_false_key_beside_others_loads(tmp_path, other):
    """O-A2 C1 (review B; the mutant refuses whenever the key is set, ``is not None`` for ``is True``): a live
    profile with ``overview_frame = false`` beside ``{other} = true`` loads, keeping both values; the same profile with
    ``overview_frame = true`` is refused (the discriminating pair)."""
    require()
    active = _load(
        tmp_path, f"false-{other}", f'tools = "live"\noverview_frame = false\n{other} = true\n'
    ).active()
    assert active.overview_frame is False and getattr(active, other) is True, active  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        _load(tmp_path, f"true-{other}", f'tools = "live"\noverview_frame = true\n{other} = true\n')


def test_line_placement(tmp_path, capsys, monkeypatch):
    """A2-9: exactly one line, right after the ``auto overview`` line, under ``--dry-run`` and at a real start (and
    there before the regime line); none in ``off`` or ``dry_run``. T-0152 re-pin R2: the real start's default line
    reads ``on (default)``."""
    require()
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", ON)
    ls = out.splitlines()
    assert lines(out) == [PREFIX + "on (flag)"], out
    (a,) = [i for i, x in enumerate(ls) if x.startswith(AUTO_PREFIX)]
    assert ls[a + 1] == PREFIX + "on (flag)", out
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "s", capsys, monkeypatch, *LIVE)
    assert code == 0, started
    sl = started.splitlines()
    assert lines(started) == [PREFIX + "on (default)"], started  # T-0152 re-pin R2
    (a,) = [i for i, x in enumerate(sl) if x.startswith(AUTO_PREFIX)]
    (r,) = [i for i, x in enumerate(sl) if x.startswith("astra session: regime:")]
    assert sl[a + 1] == PREFIX + "on (default)" and a + 1 < r, started  # T-0152 re-pin R2
    for mode in ("off", "dry_run"):
        monkeypatch.undo()
        assert "overview frame:" not in dry_run(tmp_path / mode, capsys, monkeypatch, mode)


def test_profile_line(tmp_path, capsys, monkeypatch):
    """A2-9: a live profile's ``overview_frame = true`` gives ``on (profile)`` and the keyword."""
    require()
    out = dry_run(
        tmp_path / "d", capsys, monkeypatch, "live", body=LIVE_PROFILE + "overview_frame = true\n"
    )
    assert lines(out) == [PREFIX + "on (profile)"], out


# =====================================================================================================
# startup refusals
# =====================================================================================================


@pytest.mark.parametrize("mode", ["off", "dry_run"])
@pytest.mark.parametrize("flag", [ON, OFF])
def test_flag_refused_outside_live(tmp_path, capsys, monkeypatch, mode, flag):
    """A2-9: either flag in a resolved mode other than live exits 2 naming ``live`` and builds nothing (also through a
    live profile whose mode ``--tools`` overrode); checked after the flags exist, so argparse's exit 2 cannot satisfy
    it."""
    require()
    for flags, body in (
        (MODE_FLAGS[mode], ""),
        (["--tools", "off" if mode == "off" else "dry-run"], LIVE_PROFILE),
    ):
        code, text, calls = refused_run(tmp_path, capsys, monkeypatch, *flags, flag, body=body)
        assert code == 2 and _names_live(text) and "unrecognized" not in text, (code, text)
        assert not calls.graphs and not calls.session_kwargs
        monkeypatch.undo()


@pytest.mark.parametrize("override", [["--tools", "off"], ["--tools", "dry-run"]])
def test_key_refused_when_tools_overridden(tmp_path, capsys, monkeypatch, override):
    """A2-9: a live profile with an ``overview_frame`` key under a non-live ``--tools`` override exits 2 naming
    ``live`` and builds nothing, with ``--dry-run`` and without; too-wide companion: no override builds."""
    require()
    body = LIVE_PROFILE + "overview_frame = true\n"
    for dry in (True, False):
        code, text, calls = refused_run(
            tmp_path, capsys, monkeypatch, *override, body=body, dry=dry
        )
        assert code == 2 and _names_live(text), (override, dry, code, text)
        assert not calls.graphs and not calls.session_kwargs
        monkeypatch.undo()
    calls, _ = build(tmp_path / "ok", capsys, monkeypatch, body=body)
    assert calls.graphs


#: the other mode, its flag, its profile key line and the words that name it
OTHERS = {
    "auto": ("--auto-overview", "--no-auto-overview", "auto_overview = true\n", "auto overview"),
    "page": ("--page-overview", "--no-page-overview", "page_overview = true\n", "page overview"),
    "reground": ("--reground", "--no-reground", "reground = true\n", "reground"),
}
PAIRS = ("flag/flag", "flag/key", "key/flag")


@pytest.mark.parametrize("pair_", PAIRS)
@pytest.mark.parametrize("other", list(OTHERS))
def test_refused_with_other_overview_or_reground(tmp_path, capsys, monkeypatch, other, pair_):
    """A2-9 / A2-1 / Q-4 (reviewer mutation 15): the overview frame on together with the auto overview, the page
    overview or ReGround, from each source pair (``flag/flag``, ``flag/key``, ``key/flag``; key/key is a load
    refusal), exits 2 naming both before anything is constructed, with ``--dry-run`` and at a real start. Too-wide
    companions: ``--overview-frame`` with the other's ``--no-...`` builds, and the other's flag with
    ``--no-overview-frame`` builds."""
    require()
    on_flag, off_flag, key_line, words = OTHERS[other]
    flags, body = {
        "flag/flag": ([ON, on_flag], ""),
        "flag/key": ([ON], key_line),
        "key/flag": ([on_flag], "overview_frame = true\n"),
    }[pair_]
    for dry in (True, False):
        code, text, calls = refused_run(
            tmp_path, capsys, monkeypatch, *LIVE, *flags, body=LIVE_PROFILE + body, dry=dry
        )
        assert code == 2 and names_both(text, words), (other, pair_, dry, code, text)
        assert "unrecognized" not in text
        assert not calls.graphs and not calls.session_kwargs, "built before refusing"
        monkeypatch.undo()
    for ok in ([ON, off_flag], [on_flag, OFF]):
        calls, _ = build(tmp_path / ("ok" + "".join(ok)), capsys, monkeypatch, *ok)
        assert calls.graphs, ok
        monkeypatch.undo()
