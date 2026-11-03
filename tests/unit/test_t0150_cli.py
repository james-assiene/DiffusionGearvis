"""T-0150 A6 (cli, config): the opt-in, its source, the refusals, the wiring and the startup line
(docs/tasks/T-0150.md).

**The surface these rows pin** (the task's names; readings stated):

- ``config.LIVE_DEFAULT_AUTO_OVERVIEW = False``; ``Profile.auto_overview: bool | None = None``; a non-bool key is a load
  refusal naming true and false (the ``page_overview_refusal`` pattern); the key with ``tools != "live"`` is a load
  refusal naming ``live``; ``auto_overview = true`` with ``page_overview = true`` is a load refusal naming both.
- ``astra session --auto-overview`` / ``--no-auto-overview``; argparse dest ``args.auto_overview`` (read with
  ``getattr``). ``cli.auto_overview_of(profile, args, mode) -> AutoOverview(auto_overview, auto_overview_from)``,
  read here as a two-field value in that order (``pair``, T-0140's reading); sources ``flag``/``profile``/``default``;
  outside live only "not on" is pinned.
- Startup refusals exit 2 before anything is constructed: the flag outside live (naming ``live``), the key of a live
  profile under a non-live ``--tools`` override, and auto plus page overview on from any source pair (naming both:
  ``auto`` and ``page`` overview, read as the two words "auto" and "page" in the message).
- ``Session`` gets ``auto_overview=True`` only when on; off, the keyword is absent. The kit stays v1 (bbfc53cc...).
- The line, live only: ``astra session: auto overview: on (flag)`` / ``off (default)`` / ``on (profile)``, right after
  the ``page overview`` line and before the regime line (``--dry-run`` too; there is no regime line under
  ``--dry-run``, so there only "right after" is pinned).

Regime: CPU, the cli rigs over fakes, nothing served, n = 1.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

import pytest

from local_astra import cli, config
from local_astra.config import Profile, load
from tests.unit import _t0202_pins as t0202
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_t0131 import MODE_FLAGS, dry_run
from tests.unit.test_cli_t0132 import _toml, build
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0134c_live_feedback_default import _names_live
from tests.unit.test_t0140_cli import pair, session_help

V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
PREFIX = "astra session: auto overview: "
PAGE_PREFIX = "astra session: page overview: "
ON, OFF = "--auto-overview", "--no-auto-overview"
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'
LIVE = MODE_FLAGS["live"]


def require() -> Any:
    flags = set(re.findall(r"--[a-z][a-z-]+", session_help()))
    missing = [f for f in (ON, OFF) if f not in flags]
    assert not missing, f"T-0150 A6: astra session {missing} do not exist yet"
    assert "auto_overview" in Profile.model_fields, (
        "T-0150 A6: Profile.auto_overview does not exist yet"
    )
    fn = getattr(cli, "auto_overview_of", None)
    assert callable(fn), "T-0150 A6: cli.auto_overview_of(profile, args, mode) does not exist yet"
    return fn


def lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith(PREFIX)]


def names_both(text: str) -> bool:
    low = text.lower()
    return "auto" in low and "page" in low and "overview" in low


def refused_run(
    tmp_path: Any, capsys: Any, monkeypatch: Any, *flags: str, body: str = "", dry: bool = True
) -> tuple[int, str, T.Calls]:
    monkeypatch.setenv("DISPLAY", ":99")
    calls = T.Calls()
    T.fake_clients(monkeypatch, calls)
    T.stop_before_serving(monkeypatch, calls)
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    fake_live(monkeypatch, Events())
    code = run_cli_argv(argv(tmp_path, *flags, config_body=body), dry)
    out, err = _out(capsys)
    return code, out + err, calls


def run_cli_argv(args: list[str], dry: bool) -> int:
    from tests.unit.test_cli_session import run_cli

    return int(run_cli([*args, "--dry-run"] if dry else args))


# =====================================================================================================
# config
# =====================================================================================================


def test_default_off(tmp_path, capsys, monkeypatch):
    """A6 (reviewer mutation 6, the default on): the constant is ``False``; the key defaults to ``None`` (annotation
    ``bool | None``) in every built-in profile; the resolver says ``(False, "default")``; the default build passes no
    ``auto_overview`` keyword; the default live line reads ``off (default)``."""
    fn = require()
    assert getattr(config, "LIVE_DEFAULT_AUTO_OVERVIEW", None) is False
    f = Profile.model_fields["auto_overview"]
    assert f.default is None and f.annotation == (bool | None), f
    for name, p in load(None).profiles.items():
        assert p.auto_overview is None, name  # type: ignore[attr-defined]
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    assert pair(fn(profile, argparse.Namespace(), "live")) == (False, "default")
    calls, _ = build(tmp_path / "b", capsys, monkeypatch)
    assert "auto_overview" not in calls.session_kwargs[-1], (
        "T-0150 A6: off builds call Session exactly as at BASE (no keyword)"
    )
    monkeypatch.undo()
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live")
    assert lines(out) == [PREFIX + "off (default)"], out


def test_profile_key_load_refusals(tmp_path):
    """A6: the key with ``tools`` other than ``live`` (off, dry_run, absent) is refused at load naming ``live``; a
    non-bool key is refused naming true and false; ``auto_overview = true`` with ``page_overview = true`` is refused
    naming both. Too-wide companions: each key alone, and true with ``page_overview = false``, load."""
    require()
    for tools_line in ('tools = "off"\n', 'tools = "dry_run"\n', ""):
        for v in ("true", "false"):
            d = tmp_path / f"t{len(tools_line)}-{v}"
            d.mkdir()
            with pytest.raises(ValueError) as exc:
                load(_toml(d, f"{tools_line}auto_overview = {v}\n"))
            assert _names_live(str(exc.value)), (tools_line, v, str(exc.value))
    for junk in ('"on"', '"true"', "1", "0", '"yes"'):
        d = tmp_path / f"junk-{junk.strip(chr(34))}"
        d.mkdir()
        with pytest.raises(ValueError) as exc:
            load(_toml(d, f'tools = "live"\nauto_overview = {junk}\n'))
        low = str(exc.value).lower()
        assert "true" in low and "false" in low, (junk, str(exc.value))
    d = tmp_path / "both"
    d.mkdir()
    with pytest.raises(ValueError) as exc:
        load(_toml(d, 'tools = "live"\nauto_overview = true\npage_overview = true\n'))
    assert "auto_overview" in str(exc.value) and "page_overview" in str(exc.value), str(exc.value)
    for body, want in (
        ('tools = "live"\nauto_overview = true\n', True),
        ('tools = "live"\nauto_overview = false\n', False),
        ('tools = "live"\nauto_overview = true\npage_overview = false\n', True),
        ('tools = "live"\nauto_overview = false\npage_overview = true\n', False),
    ):
        d = tmp_path / f"ok-{abs(hash(body)) % 10_000}"
        d.mkdir()
        assert load(_toml(d, body)).active().auto_overview is want, body  # type: ignore[attr-defined]


# =====================================================================================================
# the resolver, the wiring, the line
# =====================================================================================================

SOURCES = {
    "flag-on": ([ON], "", (True, "flag")),
    "flag-off": ([OFF], "", (False, "flag")),
    "profile-true": ([], "auto_overview = true\n", (True, "profile")),
    "profile-false": ([], "auto_overview = false\n", (False, "profile")),
    "flag-off-over-profile": ([OFF], "auto_overview = true\n", (False, "flag")),
    "flag-on-over-profile": ([ON], "auto_overview = false\n", (True, "flag")),
}


@pytest.mark.parametrize("case", list(SOURCES))
def test_sources_and_wiring(tmp_path, capsys, monkeypatch, case):
    """A6: flag, else key, else default, each with its source; the dry-run line says it; the built ``Session`` gets
    ``auto_overview=True`` only when on (absent when off); the kit's ``tools_version`` stays v1's either way."""
    fn = require()
    flags, body, (value, src) = SOURCES[case]
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body)
    assert lines(out) == [PREFIX + ("on" if value else "off") + f" ({src})"], out
    monkeypatch.undo()
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    kw = calls.session_kwargs[-1]
    if value:
        assert kw.get("auto_overview") is True, (
            f"T-0150 A6: Session(auto_overview=True); {sorted(kw)}"
        )
    else:
        assert "auto_overview" not in kw, f"T-0150 A6: off passes no keyword; {sorted(kw)}"
    # T-0202 D3 re-pin: the tools stay the default live set (v1, the wait set once the wait tool is on by default)
    assert calls.kit().tools_version == t0202.default_live_sha(), (
        "T-0150 A6: the tools stay v1 with auto overview on"
    )
    assert t0202.default_on() or calls.kit().tools_version == V1_SHA
    (tmp_path / "p").mkdir()
    profile = load(_toml(tmp_path / "p", LIVE_PROFILE + body)).active()
    flag = True if ON in flags else False if OFF in flags else None
    assert pair(fn(profile, argparse.Namespace(auto_overview=flag), "live")) == (value, src)


def test_resolver_not_on_outside_live() -> None:
    """A6: outside a resolved mode of live the resolver is never on (only "not on" is pinned)."""
    fn = require()
    for mode in ("off", "dry_run"):
        value, _ = pair(fn(Profile(name="lean"), argparse.Namespace(auto_overview=None), mode))
        assert not value, mode


def test_line_placement(tmp_path, capsys, monkeypatch):
    """A6: exactly one line, right after the ``page overview`` line, under ``--dry-run`` and at a real start (and
    there before the regime line); none in ``off`` or ``dry_run``."""
    require()
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", ON)
    ls = out.splitlines()
    assert lines(out) == [PREFIX + "on (flag)"], out
    (p,) = [i for i, x in enumerate(ls) if x.startswith(PAGE_PREFIX)]
    assert ls[p + 1] == PREFIX + "on (flag)", out
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "s", capsys, monkeypatch, *LIVE)
    assert code == 0, started
    sl = started.splitlines()
    assert lines(started) == [PREFIX + "off (default)"], started
    (p,) = [i for i, x in enumerate(sl) if x.startswith(PAGE_PREFIX)]
    (r,) = [i for i, x in enumerate(sl) if x.startswith("astra session: regime:")]
    assert sl[p + 1] == PREFIX + "off (default)" and p + 1 < r, started
    for mode in ("off", "dry_run"):
        monkeypatch.undo()
        assert "auto overview:" not in dry_run(tmp_path / mode, capsys, monkeypatch, mode)


# =====================================================================================================
# startup refusals
# =====================================================================================================


@pytest.mark.parametrize("mode", ["off", "dry_run"])
@pytest.mark.parametrize("flag", [ON, OFF])
def test_flag_refused_outside_live(tmp_path, capsys, monkeypatch, mode, flag):
    """A6: either flag in a resolved mode other than live exits 2 naming ``live`` and builds nothing (also through a
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
    """A6 (O-A2 Q7): a live profile with an ``auto_overview`` key under a non-live ``--tools`` override exits 2 naming
    ``live`` and builds nothing, with ``--dry-run`` and without; too-wide companion: no override builds."""
    require()
    body = LIVE_PROFILE + "auto_overview = true\n"
    for dry in (True, False):
        code, text, calls = refused_run(
            tmp_path, capsys, monkeypatch, *override, body=body, dry=dry
        )
        assert code == 2 and _names_live(text), (override, dry, code, text)
        assert not calls.graphs and not calls.session_kwargs
        monkeypatch.undo()
    calls, _ = build(tmp_path / "ok", capsys, monkeypatch, body=body)
    assert calls.graphs


PAIRS = {
    "flag/flag": ([ON, "--page-overview"], ""),
    "flag/key": ([ON], "page_overview = true\n"),
    "key/flag": (["--page-overview"], "auto_overview = true\n"),
}


@pytest.mark.parametrize("case", list(PAIRS))
def test_auto_and_page_overview_refused(tmp_path, capsys, monkeypatch, case):
    """A6 / H-T150-2 (reviewer mutation 9): the auto overview and the page overview, both on from any source pair,
    exit 2 naming both before anything is constructed (``--dry-run`` and a real start). Too-wide companions: the flag
    of one with ``--no-...`` of the other builds."""
    require()
    flags, body = PAIRS[case]
    for dry in (True, False):
        code, text, calls = refused_run(
            tmp_path, capsys, monkeypatch, *LIVE, *flags, body=LIVE_PROFILE + body, dry=dry
        )
        assert code == 2 and names_both(text), (case, dry, code, text)
        assert not calls.graphs and not calls.session_kwargs, "built before refusing"
        monkeypatch.undo()
    for ok in ([ON, "--no-page-overview"], ["--page-overview", OFF]):
        calls, _ = build(tmp_path / ("ok" + "".join(ok)), capsys, monkeypatch, *ok)
        assert calls.graphs, ok
        monkeypatch.undo()
