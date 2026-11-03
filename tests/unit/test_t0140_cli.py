"""T-0140 A5 (cli side): the opt-in, its source, the kit it builds, the line and the tools line (docs/tasks/T-0140.md).

**The surface these rows pin** (the task's; readings reported):

- ``config.LIVE_DEFAULT_PAGE_OVERVIEW = False``; ``Profile.page_overview: bool | None = None``; a non-``None`` key with
  ``tools`` other than ``live`` is a load refusal naming ``live``.
- ``astra session --page-overview`` / ``--no-page-overview``; either one in a resolved mode other than ``live`` exits 2
  naming ``live`` (not argparse's usage error) and builds nothing.
- ``cli.page_overview_of(profile, args, mode)``: **reading** the result is a two-field value ``(page_overview, source)``
  (a dataclass in that field order, the ``Recovery`` shape, or a 2-tuple); sources ``flag`` / ``profile`` /
  ``default``; the flag's argparse dest is ``args.page_overview`` (``True``/``False``/``None``), read with ``getattr``.
  Outside live only "not on" is pinned (``None`` or ``False``).
- When on, the live kit ``cli`` passes to ``Session`` has ``tools``/``tools_version`` from ``tools_v1_overview.json``;
  when off, v1's (bbfc53cc…).
- The line, live only, after T-0134c's ``feedback`` line (the recovery line already sits right after it, T-0135b, so
  only "after" is pinned): ``astra session: page overview: on (flag)`` / ``off (default)``; ``(profile)`` by the
  ``GroundingSource`` spelling.
- The ``--dry-run`` tools line shows the selected set's sha12.

O-A2 rulings applied: Q5 sources ``flag``/``profile``/``default``; Q6 after feedback, before regime; Q7 the key under a
non-live override exits 2 naming live; Q8 a non-bool key is a load refusal. Activated: the A2 rows are red until A2 is
implemented. Regime: CPU, the cli rigs over fakes, nothing served, n = 1.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import re
from pathlib import Path
from typing import Any

import pytest

from local_astra import cli, config
from local_astra.config import Profile, load
from local_astra.toolcall import native, tools
from tests.unit import _t0160_pins as t0160
from tests.unit import _t0202_pins as t0202
from tests.unit import _tools_switch_support as T
from tests.unit.test_cli_session import CLI, flags_of, run_cli
from tests.unit.test_cli_t0131 import BASE_LINES, MODE_FLAGS, SHA12, dry_run, tools_lines
from tests.unit.test_cli_t0132 import _toml, build
from tests.unit.test_cli_tools_live import Events, _out, argv, fake_live
from tests.unit.test_t0132_live_default import build_out
from tests.unit.test_t0134c_live_feedback_default import FEEDBACK_PREFIX, _names_live

V1_SHA = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
#: tests/unit/test_t0140_tools.py's pinned literal (O-A1-1)
OV_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"
PREFIX = "astra session: page overview: "
ON, OFF = "--page-overview", "--no-page-overview"
LIVE_PROFILE = 'tools = "live"\nbrowser = "headless"\n'
LIVE = MODE_FLAGS["live"]


def session_help() -> str:
    """``astra session --help`` (argparse's own text, so a ``BooleanOptionalAction`` pair counts as both flags)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        run_cli(["session", "--help"])
    return buf.getvalue()


def require() -> Any:
    flags = set(flags_of(CLI)) | set(re.findall(r"--[a-z][a-z-]+", session_help()))
    missing = [f for f in (ON, OFF) if f not in flags]
    assert not missing, f"T-0140 A5: astra session {missing} do not exist yet"
    assert "page_overview" in Profile.model_fields, (
        "T-0140 A5: Profile.page_overview does not exist yet"
    )
    fn = getattr(cli, "page_overview_of", None)
    assert callable(fn), "T-0140 A5: cli.page_overview_of(profile, args, mode) does not exist yet"
    return fn


def pair(v: Any) -> tuple[Any, Any]:
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        fs = dataclasses.fields(v)
        assert len(fs) == 2, (
            f"reading: a two-field (value, source) result; got {[f.name for f in fs]}"
        )
        return getattr(v, fs[0].name), getattr(v, fs[1].name)
    assert isinstance(v, tuple) and len(v) == 2, v
    return v[0], v[1]


def lines(out: str) -> list[str]:
    return [x for x in out.splitlines() if x.startswith(PREFIX)]


def kit_judge_set(calls: T.Calls) -> Any:
    """A6: the cli-built kit's tool-set field (the one the judge and the overview offer read), found by its type."""
    from tests.unit.test_t0140_session import kit_set_field

    return getattr(calls.kit(), kit_set_field())


def kit_version(calls: T.Calls) -> str:
    kit = calls.kit()
    assert kit is not None, "premise: a kit was built"
    return str(kit.tools_version)


def test_default_off(tmp_path, capsys, monkeypatch):
    """A5 (reviewer mutation 5, the overview offered by default): the constant is ``False``, the profile key defaults
    to ``None`` everywhere, the resolver says ``(False, "default")``, the built live kit is v1 (tools and sha), and the
    default line reads ``off (default)``."""
    fn = require()
    assert getattr(config, "LIVE_DEFAULT_PAGE_OVERVIEW", None) is False
    f = Profile.model_fields["page_overview"]
    assert f.default is None and f.annotation == (bool | None), f
    for name, p in load(None).profiles.items():
        assert p.page_overview is None, name  # type: ignore[attr-defined]
    profile = Profile(name="lean", tools="live")  # type: ignore[arg-type]
    assert pair(fn(profile, argparse.Namespace(), "live")) == (False, "default")
    # T-0202 D3 re-pin: the wait tool pinned off, so "no overview by default" reads as BASE's v1 kit (the default's
    # wait set is T-0202 D's row; this row pins only that the overview set is not the default)
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *t0202.pin_off([]))
    kit = calls.kit()
    assert kit.tools_version == V1_SHA and kit.tools == tuple(native.tools_field(tools.load())), (
        "T-0140 A5: the default live kit is v1"
    )
    assert kit_judge_set(calls) is None, (
        f"T-0140 A6 (review B F1): the default kit's tool set is None (v1); got {kit_judge_set(calls)!r}"
    )
    monkeypatch.undo()
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live")
    assert lines(out) == [PREFIX + "off (default)"], out


SOURCES = {
    "flag-on": ([ON], "", (True, "flag")),
    "flag-off": ([OFF], "", (False, "flag")),
    "profile-true": ([], "page_overview = true\n", (True, "profile")),
    "profile-false": ([], "page_overview = false\n", (False, "profile")),
    "flag-off-over-profile": ([OFF], "page_overview = true\n", (False, "flag")),
    "flag-on-over-profile": ([ON], "page_overview = false\n", (True, "flag")),
}


@pytest.mark.parametrize("case", list(SOURCES))
# T-0160 ruling 2: web on by default in live (egress=web)
def test_sources(tmp_path, capsys, monkeypatch, case):
    """A5: flag, else key, else default, each with its source; the line says it; the kit cli builds (``tools`` and
    ``tools_version``) is the overview set's when on and v1's when off; the dry-run tools line carries that sha12."""
    fn = require()
    flags, body, (value, src) = SOURCES[case]
    # T-0202 D3 re-pin: the wait tool pinned off, so the sets are v1 and overview (B8's wait combinations are
    # test_t0202_cli.test_flag_selects_set's)
    flags = t0202.pin_off(flags, body)
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", *flags, body=LIVE_PROFILE + body)
    assert lines(out) == [PREFIX + ("on" if value else "off") + f" ({src})"], out
    sha = OV_SHA if value else V1_SHA
    # T-0160 pin move (C-2): the default live line reads egress=web
    want_line = t0160.web_line(BASE_LINES["live"]).replace(SHA12, sha[:12]) + " visual_tokens=1120"
    assert tools_lines(out) == [want_line], (
        f"T-0140 A5: the tools line shows the set's sha12\n{out}"
    )
    monkeypatch.undo()
    calls, _ = build(tmp_path / "b", capsys, monkeypatch, *flags, body=LIVE_PROFILE + body)
    assert kit_version(calls) == sha, (case, kit_version(calls))
    want_set = tools.load(tools.TOOLS_V1_OVERVIEW_PATH) if value else tools.load()
    assert calls.kit().tools == tuple(native.tools_field(want_set)), case
    judged = kit_judge_set(calls)
    if value:
        assert judged is not None and judged.version == OV_SHA, (
            f"T-0140 A6 (review B F1) {case}: the kit carries the overview set its judge reads; got {judged!r}"
        )
        assert "page_overview" in judged.names, judged.names
    else:
        assert judged is None, (
            f"T-0140 A6 (review B F1) {case}: off, the kit's tool set is None; got {judged!r}"
        )
    (tmp_path / "p").mkdir()
    profile = load(_toml(tmp_path / "p", LIVE_PROFILE + body)).active()
    flag = True if ON in flags else False if OFF in flags else None
    assert pair(fn(profile, argparse.Namespace(page_overview=flag), "live")) == (value, src)


def test_line_after_feedback(tmp_path, capsys, monkeypatch):
    """A5: exactly one line, after the ``feedback`` line, under ``--dry-run`` and at a real start (and there before
    the regime line); none in ``off`` or ``dry_run``."""
    require()
    out = dry_run(tmp_path / "d", capsys, monkeypatch, "live", ON)
    ls = out.splitlines()
    assert lines(out) == [PREFIX + "on (flag)"], out
    (f,) = [i for i, x in enumerate(ls) if x.startswith(FEEDBACK_PREFIX)]
    assert ls.index(PREFIX + "on (flag)") > f, out
    monkeypatch.undo()
    _, started, code = build_out(tmp_path / "s", capsys, monkeypatch, *LIVE)
    assert code == 0, started
    sl = started.splitlines()
    assert lines(started) == [PREFIX + "off (default)"], started
    (f,) = [i for i, x in enumerate(sl) if x.startswith(FEEDBACK_PREFIX)]
    (r,) = [i for i, x in enumerate(sl) if x.startswith("astra session: regime:")]
    assert f < sl.index(PREFIX + "off (default)") < r, started
    for mode in ("off", "dry_run"):
        monkeypatch.undo()
        assert "page overview:" not in dry_run(tmp_path / mode, capsys, monkeypatch, mode)


@pytest.mark.parametrize("mode", ["off", "dry_run"])
@pytest.mark.parametrize("flag", [ON, OFF])
def test_flag_refused_outside_live(tmp_path, capsys, monkeypatch, mode, flag):
    """A5: either flag in a resolved mode other than live exits 2 naming ``live`` and builds nothing (also through a
    live profile whose mode ``--tools`` overrode); checked after the flags exist, so argparse's own exit 2 cannot
    satisfy it."""
    require()
    for flags, body in (
        (MODE_FLAGS[mode], ""),
        (["--tools", "off" if mode == "off" else "dry-run"], LIVE_PROFILE),
    ):
        monkeypatch.setenv("DISPLAY", ":99")
        calls = T.Calls()
        T.fake_clients(monkeypatch, calls)
        T.stop_before_serving(monkeypatch, calls)
        T.patch_detok_load(monkeypatch, calls, tmp_path)
        fake_live(monkeypatch, Events())
        code = run_cli([*argv(tmp_path, *flags, flag, config_body=body), "--dry-run"])
        out, err = _out(capsys)
        assert code == 2 and _names_live(out + err) and "unrecognized" not in err, (code, out, err)
        assert not calls.graphs and not calls.session_kwargs
        monkeypatch.undo()


def test_profile_key_load_refusal(tmp_path):
    """A5: the key (true or false) with ``tools`` other than ``live`` (off, dry_run, or no tools key) is refused at
    load naming ``live``; with ``tools = "live"`` both load and read back."""
    require()
    for tools_line in ('tools = "off"\n', 'tools = "dry_run"\n', ""):
        for v in ("true", "false"):
            d = tmp_path / f"{tools_line[9:12]}-{v}"
            d.mkdir()
            with pytest.raises(ValueError) as exc:
                load(_toml(d, f"{tools_line}page_overview = {v}\n"))
            assert _names_live(str(exc.value)), (tools_line, v, str(exc.value))
    for v, want in (("true", True), ("false", False)):
        d = tmp_path / f"ok-{v}"
        d.mkdir()
        assert (
            load(_toml(d, f'tools = "live"\npage_overview = {v}\n')).active().page_overview is want
        )  # type: ignore[attr-defined]


@pytest.mark.parametrize("override", [["--tools", "off"], ["--tools", "dry-run"]])
@pytest.mark.parametrize("value", ["true", "false"])
def test_profile_key_refused_when_tools_overridden(tmp_path, capsys, monkeypatch, override, value):
    """O-A2 Q7 (T-0134c's rule): a live profile with a ``page_overview`` key run under a non-live ``--tools`` override
    exits 2 naming ``live`` and builds nothing, with ``--dry-run`` and without. Too-wide companion: the same profile
    with no override builds."""
    require()
    body = LIVE_PROFILE + f"page_overview = {value}\n"
    for extra in (["--dry-run"], []):
        monkeypatch.setenv("DISPLAY", ":99")
        calls = T.Calls()
        T.fake_clients(monkeypatch, calls)
        T.stop_before_serving(monkeypatch, calls)
        T.patch_detok_load(monkeypatch, calls, tmp_path)
        fake_live(monkeypatch, Events())
        code = run_cli([*argv(tmp_path, *override, config_body=body), *extra])
        out, err = _out(capsys)
        assert code == 2 and _names_live(out + err), (override, value, extra, code, out, err)
        assert not calls.graphs and not calls.session_kwargs, "a refused run built a graph"
        monkeypatch.undo()
    calls, _ = build(tmp_path / "ok", capsys, monkeypatch, body=body)
    assert calls.graphs, "the same profile without an override is accepted"


def test_profile_key_non_bool_refused(tmp_path):
    """O-A2 Q8 (``thought_call_recovery``'s rule): a non-bool ``page_overview`` (strings and integers a lax bool would
    coerce) is refused at load, naming true and false."""
    require()
    for junk in ('"on"', '"true"', '"off"', "1", "0", '"yes"'):
        d = tmp_path / f"junk-{junk.strip(chr(34))}"
        d.mkdir()
        with pytest.raises(ValueError) as exc:
            load(_toml(d, f'tools = "live"\npage_overview = {junk}\n'))
        low = str(exc.value).lower()
        assert "true" in low and "false" in low, (junk, str(exc.value))


def test_tools_line_helper_unchanged() -> None:
    """A5/A12 (green at BASE): ``cli.tools_line`` over a v1 kit is BASE's string (it prints ``tools_version[:12]``)."""
    live = cli.LiveLine(browser="headless", viewport=(1280, 800))
    assert cli.tools_line(T.make_kit(), 8, live) == BASE_LINES["live"]
    assert V1_SHA[:12] == SHA12
    assert Path(tools.TOOLS_V1_OVERVIEW_PATH).is_file()
